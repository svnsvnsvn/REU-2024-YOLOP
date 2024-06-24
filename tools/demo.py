import argparse  # For parsing command line arguments
import os, sys  # For interacting with the operating system and system path
import shutil  # For high-level file operations like copying and deleting
import time  # For time-related functions
from pathlib import Path  # For object-oriented file path manipulation
import imageio  # For reading and writing image data

# Set the base directory and add it to the system path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2  # OpenCV library for image and video processing
import torch  # PyTorch library for tensor computation and deep learning
import torch.backends.cudnn as cudnn  # For CUDA backend optimizations
from numpy import random  # For generating random numbers
import scipy.special  # For special mathematical functions
import numpy as np  # For numerical operations on arrays
import torchvision.transforms as transforms  # For image transformations
import PIL.Image as image  # For image handling

# Import custom modules and functions
from lib.config import cfg  # Configuration settings
from lib.config import update_config  # Function to update configuration
from lib.utils.utils import create_logger, select_device, time_synchronized  # Utility functions
from lib.models import get_net  # Function to get the neural network model
from lib.dataset import LoadImages, LoadStreams  # Functions to load images or video streams
from lib.core.general import non_max_suppression, scale_coords  # General-purpose functions
from lib.utils import plot_one_box, show_seg_result  # Utility functions for plotting and displaying results
from lib.core.function import AverageMeter  # Utility for averaging measurements
from lib.core.postprocess import morphological_process, connect_lane  # Post-processing functions
from tqdm import tqdm  # Progress bar library

# Normalization parameters for the input images
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

# Transformation pipeline for the input images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    normalize,  # Normalize the tensor
])

def add_noise(img, noise_level=0.1):
    """ Add random noise to an image tensor """
    noise = torch.randn(img.size()) * noise_level
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0, 1)  # Ensure values are in [0, 1] range

def detect(cfg, opt):
    # Create logger for logging information
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')

    # Select device (CPU or GPU) for computation
    device = select_device(logger, opt.device)
    print("the device is " + device.type)
    if os.path.exists(opt.save_dir):  # Check if output directory exists
        shutil.rmtree(opt.save_dir)  # Delete directory if exists
    os.makedirs(opt.save_dir)  # Create new output directory
    half = device.type != 'cpu'  # Half precision only supported on CUDA

    # Load the model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # Convert model to FP16

    # Set up data loader: is it coming from a directory with images or from  a camera?
    if opt.source.isnumeric():
        cudnn.benchmark = True  # Speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # Batch size 
    else: # Coming from a video source
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # process 1 frame or image at a time

    # Get names and colors for visualization
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # Initialize image
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # Run once to initialize
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    # # Process each image or frame in the dataset
    # for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
    #     img = transform(img).to(device)
    #     img = img.half() if half else img.float()  # Convert image to appropriate precision
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    
    # Process each image or frame in the dataset
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        img = transform(img).to(device)
       # img = add_noise(img)  # Add noise to the image tensor
        img = img.half() if half else img.float()  # Convert image to appropriate precision
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out, ll_seg_out = model(img)
        t2 = time_synchronized()
        
        inf_out, _ = det_out
        inf_time.update(t2 - t1, img.size(0))

        # Apply Non-Max Suppression (NMS)
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4 - t3, img.size(0))
        det = det_pred[0]

        # Set save path for the output
        save_path = str(opt.save_dir + '/' + Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        # Get image dimensions and padding
        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        # Process segmentation output for the drive area (da_seg_out)
        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        
        # Process segmentation output for lane lines (ll_seg_out)
        ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        
        # Visualize the segmentation results on the image
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        # Draw bounding boxes on the image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_det.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det, label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
        # Save the results
        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_det)
        elif dataset.mode == 'video':
            if vid_path != save_path:  # New video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # Release previous video writer

                fourcc = 'mp4v'  # Output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h, w, _ = img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    # Print final results
    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('Inference time: (%.4fs/frame)   NMS time: (%.4fs/frame)' % (inf_time.avg, nms_time.avg))

if __name__ == '__main__':
    # Argument parser to read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    # Run the detection function with no gradient calculation
    with torch.no_grad():
        detect(cfg, opt)
