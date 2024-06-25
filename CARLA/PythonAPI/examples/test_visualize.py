import cv2, os
import numpy as np
from position2bev import pos2fv
path = '/home/reu/carla/PythonAPI/examples/Carla_Recorder'

img = cv2.imread(os.path.join(path, "semantic_Recorder/train", "5_355835.jpg"))


save_path = os.path.join(path, "tmp")
# img_out = pos2fv("5_355835.jpg", img, save=True, save_path=save_path, depth=False)

sem_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(sem_img, 17, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((0, 0), np.uint8)
blur = cv2.dilate(thresh, kernel)

contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)  
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    rects.append([x, y,x + w, y + h])

for (x1, y1, x2, y2) in rects:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite(os.path.join(path, "tmp", 'bbox.jpg'), img)