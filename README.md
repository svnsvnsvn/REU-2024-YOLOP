This research was conducted at Oakland University in the REU program and partially supported by the NSF REU grants CNS-1852475 and CNS-1938687.

Advisor For This Project: Dr. Lanyu (Lori) Xu 

Graduate Student Mentor: Yunge Li 

Undergraduate Student Researchers: Harrison Haviland-Longo, Ann Ubaka
### Abstract

Autonomous vehicles (AVs) leverage machine-learning perception models to detect and classify critical objects such as road signs, vehicles, lane lines, hazards, and pedestrians, enabling self-driving functionalities. With the nationwide proliferation of AVs, the demand for safe, secure, accurate, and rapid driving perception models has surged dramatically. Panoptic perception models have been proposed to offer advanced object detection and segmentation capabilities for AVs. This work explores the robustness of panoptic perception models against adversarial attacks, focusing on the YOLO-P (You Only Look Once for Panoptic Driving Perception) model. To comprehensively evaluate the safety of panoptic perception models, the model is subjected to various adversarial attacks, including the Fast Gradient Sign Method (FGSM), Jacobian-based Saliency Map Attack (JSMA), Color Channel Perturbations (CCP), and Universal Adversarial Perturbations (UAP), to assess their effects on model performance. Subsequent defenses, including image pre-processing techniques and the deployment of a Defense Generative Adversarial Network (GAN), are implemented to mitigate attack effects. The findings reveal deprecated performance post-attack implementation, with only marginal improvement in the performance of models post-defense application. These results suggest the necessity for further research into more effective defense mechanisms to ensure the safety and reliability of AVs against adversarial threats.

### Repository Contents

- **YOLOP Source Code**
  - Available at: [YOLOP GitHub Repository](https://github.com/hustvl/YOLOP)

- **Customized Attacks**
  - FGSM (Fast Gradient Sign Method)
  - JSMA (Jacobian-based Saliency Map Attack)
  - UAP (Universal Adversarial Perturbations)
  - CCP (Color Channel Perturbations)

- **Customized Defenses**
  - Pre-Processing Techniques
  - Defense GAN

  See `requirements.txt` for additional dependencies and version requirements.
  
  ```setup
  pip install -r requirements.txt
  ```
  
  Start evaluating:
  
  ```shell
  python tools/test.py --weights weights/End-to-end.pth --attack [FGSM|JSMA|UAP|CCP|None]
  ```   
### Additional Options

- **--modelDir:** Model directory.
- **--logDir:** Log directory (default: `runs/`).
- **--conf_thres:** Object confidence threshold (default: `0.001`).
- **--iou_thres:** IOU threshold for NMS (default: `0.6`).
- **--experiment_mode:** Experiment mode (0 or 1) (default: `1`).
  - `1 (True):` Runs with several pre-generated values.
  - `0 (False):` Provide your own parameters.

#### FGSM Attack

- **--epsilon:** Epsilon value for FGSM or CCP attack (default: `0.1`).
- **--fgsm_attack_type:** Type of FGSM attack (options: `FGSM`, `FGSM_WITH_NOISE`, `ITERATIVE_FGSM`).

#### JSMA Attack

- **--num_pixels:** Number of pixels to be perturbed after saliency calculation (default: `10`).
- **--jsma_perturbation:** Perturbation value for JSMA attack (default: `0.1`).
- **--jsma_attack_type:** Type of perturbation for JSMA attack (options: `add`, `set`, `noise`).

#### UAP Attack

- **--uap_max_iterations:** Maximum number of iterations for UAP attack (default: `10`).
- **--uap_eps:** Epsilon value for UAP attack (default: `0.03`).
- **--uap_delta:** Delta value for UAP attack (default: `0.8`).
- **--uap_num_classes:** Number of classes for UAP attack.
- **--uap_targeted:** Whether the UAP attack is targeted or not (default: `False`).
- **--uap_batch_size:** Batch size for UAP attack (default: `6`).

#### CCP Attack

- **--color_channel:** Color channel to perturb (`R`, `G`, `B`).

#### Defenses

- **--resizer:** Desired `WIDTHxHEIGHT` of your resized image.
- **--quality:** Desired quality for JPEG compression output (`0-100`).
- **--border_type:** Border type for Gaussian Blurring (`default`, `constant`, `reflect`, `replicate`).
- **--gauss:** Apply Gaussian Blurring to image. Specify `ksize` as `WIDTHxHEIGHT`.
- **--noise:** Add Gaussian Noise to image. Specify `sigma` value for noise generation.
- **--bit_depth:** Choose bit value between `1-8`.

#### Utilizing the Defense GAN

Input perturbed images of size `512x512` into the `TEST_A` folder located in the Pix2PixHD datasets folder. Once the images are loaded into the folder, run the following command to generate a new image set of synthesized images.

```shell
python test.py --dataroot /home/reu/Documents/YOLOP-main/pix2pixHD/datasets/bdd100k --name trained --netG global --label_nc 0 --no_instance --how_many number_of_images --which_epoch 200 --checkpoints_dir /home/reu/Documents/YOLOP-main/pix2pixHD/trained
```

To utilize the Defense GAN, please input perturbed images of size 512 x 512 into the TEST_A folder located in the Pix2PixHD datasets folder. Once the images are all loaded into the folder, run the following command to generate a new image set of synthesized images.

```shell
python test.py --dataroot /home/reu/Documents/YOLOP-main/pix2pixHD/datasets/bdd100k --name trained --netG global --label_nc 0 --no_instance --how_many number_of_images --which_epoch 200 --checkpoints_dir /home/reu/Documents/YOLOP-main/pix2pixHD/trained
```

After image generation, the synthesized images will be stored in a folder named after the used epoch within the "trained" subfolder of the "results" folder. After generation, a script such as "namefixer" can be used to rename all of the images after their corresponding validation image from BDD100K, and then the images can be easily inputted into YOLOP through the validation folder of BDD100K located in the YOLOP datasets folder. You may then run the YOLOP test.py script as normal to collect loss and accuracy metrics.

### Resources and Links

As part of the DEFENSE GAN training, 500 adversarial versions of images within the BDD100K training image set were generated. Due to size constraints, the images cannot be uploaded to GitHub. Access is provided via the following Google Drive links:

- [BDD100K Adversarial Images](https://drive.google.com/file/d/1GYsClGMjdcf-lCJk_mt5QLE3MYTVrFNp/view?usp=sharing)
- [Pix2Pix Image Set Based on BDD100K](https://drive.google.com/file/d/1PEuQxonaavtBCAztV1QIneYlT4sMz0qH/view?usp=sharing)
- [162 Epoch-Trained Pix2Pix](https://drive.google.com/file/d/1Dy6-QK6uJakJgegrryQNPB0pHRX8H783/view?usp=sharing)
- [200 Epoch-Trained Pix2Pix](https://drive.google.com/file/d/1y6qSAqZ95fVBVU9sj6pXv7Mh7AYQwSgz/view?usp=sharing)
- [Updated GAN Synthesized Images](https://drive.google.com/file/d/160VOgpqzqPT4Ck1oi-7G8ARRVnA_Db4j/view?usp=sharing)
- [Updated FGSM Validation Examples](https://drive.google.com/file/d/1_zsOsB6Xsz0Zjv10hRg-es44mHkU8Vd-/view?usp=sharing)

### Outside Sources

- [YOLOP GitHub Repository](https://github.com/hustvl/YOLOP)
- [BDD100K Dataset Documentation and Download](https://doc.bdd100k.com/download.html)
- [SekiroRong's YOLOP](https://github.com/SekiroRong/YOLOP)
- [NVIDIA's pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
