import carla
import numpy as np
import cv2
import torch
import sys
import os
import argparse
import pygame
import weakref
from lib.models import yolop
from lib.config import cfg
from lib.utils.utils import Prepocess, post_process
from carla import ColorConverter as cc

# YOLOP paths and setup
yolop_path = '/home/reu/YOLOP'
sys.path.append(yolop_path)
weights_path = '/home/reu/YOLOP/weights/End-to-end.pth'
device = 'cpu'  # Use CPU as specified

# Load YOLOP model
def load_model(weights_path, device):
    model = yolop.YOLOP(cfg)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

model = load_model(weights_path, device)
preprocess = Prepocess()
post_process = post_process()

def get_args():
    argparser = argparse.ArgumentParser(description="CARLA Automatic Control Client")
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name (default: "hero")')
    args = argparser.parse_args()
    return args

def visualize_results(image, det_out, da_seg_out, ll_seg_out):
    """
    Visualize YOLOP results on the input image.
    """
    # Resize to 640x640
    image = cv2.resize(image, (640, 640))

    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Visualize detection output
    for box in det_out:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, f'{cls} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Visualize drivable area segmentation
    da_seg_mask = (da_seg_out > 0.5).astype(np.uint8) * 255
    da_seg_mask = cv2.cvtColor(da_seg_mask, cv2.COLOR_GRAY2BGR)
    image_bgr = cv2.addWeighted(image_bgr, 1, da_seg_mask, 0.5, 0)

    # Visualize lane line segmentation
    ll_seg_mask = (ll_seg_out > 0.5).astype(np.uint8) * 255
    ll_seg_mask = cv2.cvtColor(ll_seg_mask, cv2.COLOR_GRAY2BGR)
    image_bgr = cv2.addWeighted(image_bgr, 1, ll_seg_mask, 0.5, 0)

    return image_bgr

def process_image(image):
    """
    Process the image using YOLOP for object detection and segmentation.
    """
    image_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_array = np.reshape(image_array, (image.height, image.width, 4))
    image_array = image_array[:, :, :3]  # Remove alpha channel
    cv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    input_image, _ = preprocess(cv_image, (640, 640))
    input_image = torch.from_numpy(input_image).unsqueeze(0).float().to(device)

    with torch.no_grad():
        results = model(input_image)

    det_out, da_seg_out, ll_seg_out = post_process(results, cfg)

    visualized_image = visualize_results(cv_image, det_out, da_seg_out, ll_seg_out)

    return visualized_image

def image_callback(image, display):
    """
    Callback function to process images from CARLA.
    """
    visualized_image = process_image(image)
    display.blit(pygame.surfarray.make_surface(visualized_image.swapaxes(0, 1)), (0, 0))
    pygame.display.flip()

def main():
    args = get_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    spawn_point = world.get_map().get_spawn_points()[0]

    vehicle_bp = blueprint_library.filter(args.filter)[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)  # Use CARLA's built-in autopilot

    # Set up pygame window
    pygame.init()
    display = pygame.display.set_mode((640, 640), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA YOLOP")

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    weak_self = weakref.ref(display)
    camera.listen(lambda image: image_callback(image, weak_self()))

    try:
        while True:
            world.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()