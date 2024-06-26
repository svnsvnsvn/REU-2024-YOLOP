#!/usr/bin/env python



#This is a very, very simple pre-processing script for image data. 
#OpenCV has documentation related to both Image-Resizing and JPEG Compression
# https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
# 


import cv2
import argparse
import os
import sys


# Image Resizer based on OpenCV's Geometric Image Transformation documentation 
def image_resizer(image, image_size=(1280, 720)):
        ''' Explanations & Arguments
        Arguments:
        image = input image
        image_size = Width x Height of output image. Defaulted at 720p
        Return:
        Resized = Resized image.
        
        Explanations:
        cv2 = OpenCV
        cv2.resize = OpenCV in-built image resizer
        Interpolation Options: 
        1. cv2.INTER_AREA = shrinking
        2. cv2.INTER_CUBIC = bicubic interpolation
        3. cv2.INTER_LINEAR = zooming
        There are more options than this... see this link: 
        Specifically in the section titled "Enumeration Type Documentation"
        https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121a55e404e7fa9684af79fe9827f36a5dc1 
        '''
        Resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        return Resized

def compress_jpg(image, image_quality=85):
        '''
        Arguments:
        image = the image.
        image_quality = Quality of the output image on a 0-100 scale, 100 would be the highest quality 0 is the highest compression.
        Return: 
        compressed = compressed image


        encode, 1 = Image read in color (BGR)
        '''
        quality_parameter = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
        #return tuple
        _, encode =  cv2.imencode('.jpg', image, quality_parameter)
        compressed = cv2.imdecode(encode, 1)
        return compressed


"""
REMAINING Pre-Processing Functions : 
1. Gaussian Blur
2. Adding Noise
3. Reducing or Removing Noise
4. Bit-Depth Reduction
"""
def main(args):
        
        file_ext = os.path.splitext(args.input_image)[1].lower()       # allow for (WIDTHxHEIGHT) args

        image = cv2.imread(args.input_image)
        if image is None: 
            raise ValueError("No image selected!")
        
        if args.resizer:       #resize 
                width, height = map(int, args.resizer.split('x'))
                image = image_resizer(image, (width, height))
                print(f"Image resized to {args.resizer}")
        
        if args.quality and file_ext not in ['.jpeg', '.jpg']:
            raise ValueError("Please do not attempt to perform JPEG compression on an image that is not a JPEG! :'( ")
        if args.quality:
            image = compress_jpg(image, args.quality)
            print(f"JPEG compressed at {args.quality}% quality")
        
        
        cv2.imwrite(args.image_output, image)

        print("Image has been pre-processed.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Image Pre-Processor Defense")
    parser.add_argument('input_image', type=str, help='Input image name, or full path.')
    parser.add_argument('image_output', type=str, help='Output image name, or full path')
    parser.add_argument('--resizer', type=str, help='Desired WIDTHxHEIGHT of your resized image')
    parser.add_argument('--quality', type=int, help='Desired quality for JPEG compression output. 0 - 100')

    args = parser.parse_args()
    main(args)
