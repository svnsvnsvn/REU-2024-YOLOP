#!/usr/bin/env python

# @Time : 2022/3/7 12:20
# @Author : 戎昱
# @File : makeTestSet.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import os
video_path = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder/testing'

images_dir = os.path.join(video_path, 'image_2')
txt_dir = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder/ImageSets/train.txt'
txt_dir2 = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder/ImageSets/test.txt'
txt_dir3 = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder/ImageSets/trainval.txt'
txt_dir4 = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder/ImageSets/val.txt'

images_filenames = sorted(
    [os.path.join(images_dir, filename) for filename in os.listdir(images_dir)]
)
with open(txt_dir2,'w') as f:
    for img in images_filenames:
        img_name = img.replace(images_dir,'')[1:-4]
        print(img_name)
        f.write(img_name)
        f.write('\n')