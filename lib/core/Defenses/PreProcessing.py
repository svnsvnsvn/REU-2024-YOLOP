# !/usr/bin/env python



#This is a simple pre-processing script for image data, allowing the user to choose which techniques to use for pre-processing.
#OpenCV has documentation related to both Image-Resizing and JPEG Compression
# https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
# 
import torch
import cv2
import argparse
import os
import sys
import numpy as np 
import matplotlib.pyplot as plt

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

def gaussian_blur(image, ksize=(5,5), sigX=5, sigY=5, borderType=cv2.BORDER_DEFAULT ):
      ''' This function allows for De-Noising and Blurring.
      Arguments:
      image = the image.
      ksize = Gaussian Kernal Size. Values must be positive and odd. (5,5) = ksize.width=5, ksize.height=5 If set to 0, compute kernal size based on sigma values.     
      sigma-X = Kernal standrad deviation for X. Controls spread of blur horizontally, with higher values meaning more blur.
      sigma-Y = Kernal standard deviation for Y. Controls spread of blur vertically, with higher values meaning more blur.
      If sigma X and sigma Y are 0 , compute sigma X-Y based upon ksize values.
      If sigma X OR sigma Y is 0, set sigma X/Y = exisitng sigma X or Y value.

      borderType = Method of extrapolating pixels:
      'BORDER_CONSTANT' = padding with a constant value
      'BORDER_REPLICATE' = pad using repetition of edge pixels
      'BORDER_REFLECT' =  pad using mirrored border pixels
      'BORDER_DEFAULT' = pad using a shift-specified reflection 
      
      kernel size=2×⌈3×σ⌉+1
      [σ] is sigma-X OR sigma-Y

      '''
      gauss = cv2.GaussianBlur(image, ksize, sigX, sigY, borderType)
      return gauss 

def noise(image, mean=0, sigma=0.2):
       ''' This function allows for the addition of randomized noise to an image.
       Information on this topic can be found here:
       https://pythonexamples.org/python-opencv-add-noise-to-image/
       https://pythontwist.com/adding-noise-to-images-in-python-with-opencv-a-simple-guide
       Arguments:
       image = the image.
       mean = Mean of noise
       sigma = Standard Deviation of noise
       '''

       noise = np.zeros_like(image) #Empty matrix w/ same shape as input image
       cv2.randn(noise, mean, sigma) #Generate randomized noise
       
       noisy = cv2.add(image, noise) #Noise is added to the image using cv2.add
       return noisy

def bit_depth(image, bits=3):
      ''' Function allows for bit-depth reduction
      Arguments:
      image = the image.
      bits = Number of bits to reduce image. This is on a scale from 1-8.

      scale = 2 ** 8
      (image // scale) * factor = Reduce image bit-depth 
      '''
      if bits < 1:
            raise ValueError("Bits must be an integer larger than one.")
      if bits > 8:
            raise ValueError("Bits must be an integer less than 8 but larger than 0")
      
      scale = 2 ** (8 - bits)
      reduction = (image // scale) * scale
      return reduction

def cv2_imshow(image):
      ''' Function allows for displaying input image and output iamge using matplot'''
      plt.imshow(cv2.cvtColor(image. cv2.COLOR_BGR2RGB))
      plt.show()
      
def process_image(image_path, args):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    if args.noise and args.gauss:
        raise ValueError("Choose either Gaussian blur OR adding noise. Gaussian blur will act as a de-noiser, rendering noise addition entirely useless.")

    if args.resizer:
        width, height = map(int, args.resizer.split('x'))
        image = image_resizer(image, (width, height))
        print(f"Image resized to {args.resizer}")

    if args.noise:
        image = noise(image, mean=0, sigma=args.noise)
        print("Noise has been applied.")

    if args.quality and not image_path.lower().endswith(('.jpeg', '.jpg')):
        raise ValueError("Please do not attempt to perform JPEG compression on a non-JPEG image.")

    if args.quality:
        image = compress_jpg(image, args.quality)
        print(f"JPEG compressed at {args.quality}% quality")

    if args.gauss:
        X, Y = map(int, args.gauss.split('x'))
        border_type = {
            'default': cv2.BORDER_DEFAULT,
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE,
        }.get(args.border_type, cv2.BORDER_DEFAULT)
        image = gaussian_blur(image, (X, Y), borderType=border_type)
        print("Applied Gaussian Blurring.")

    if args.bit_depth:
        image = bit_depth(image, bits=args.bit_depth)
        print(f"Bit depth reduced to {args.bit_depth} bits")

    if args.show:
      print("Preparing to display images...")

      ''' Display Original Image '''
      plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      plt.title('Original Image')
      plt.axis('off')
      plt.show()

      ''' Display Pre-Processsed Image '''
      plt.imshow(cv2.cvtColor(args.image_output, cv2.COLOR_BGR2RGB))
      plt.title('Processed Image')
      plt.axis('off')
      plt.show()
      
    return image

def main(args):
      if not os.path.exists(args.input_image):
            raise ValueError(f"Input directory {args.input_image} does not exist.")
      if not os.path.exists(args.image_output):
            os.makedirs(args.image_output)

      for filename in os.listdir(args.input_image):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                  input_path = os.path.join(args.input_image, filename)
                  processed_image = process_image(input_path, args)
                  # Remove "perturbed" from the filename
                  new_filename = filename.replace("_perturbed", "_defended")
                  output_path = os.path.join(args.image_output, new_filename)
                  cv2.imwrite(output_path, processed_image)
                  print(f"Processed image saved to {output_path}")

            # cv2.waitkey(10)
            # cv2.destroyAllWindows()
      print("Image has been pre-processed.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Image Pre-Processor Defense")
    parser.add_argument('--input_image', type=str, help='Input image name, or full path.')
    parser.add_argument('--image_output', type=str, help='Output image name, or full path')
    parser.add_argument('--resizer', type=str, default='1280x720', help='Desired WIDTHxHEIGHT of your resized image')
    parser.add_argument('--quality', type=int, help='Desired quality for JPEG compression output. 0 - 100')
    parser.add_argument('--border_type', type=str, choices=['default', 'constant', 'reflect', 'replicate'], default='default', help= 'border type for Gaussian Blurring')
    parser.add_argument('--gauss', type=str, help="Apply Gaussian Blurring to image. Specify ksize as WIDTHxHEIGHT")
    parser.add_argument('--noise', type=float, help='Add Gaussian Noise to image. Specify sigma value for noise generation.')
    parser.add_argument('--bit_depth', type=int, help='Choose bit value between 1 - 8')
    parser.add_argument('-s', '--show', action='store_true', help='show input image vs output image')
    args = parser.parse_args()
    main(args)
