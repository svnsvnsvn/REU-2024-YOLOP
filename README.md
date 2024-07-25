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
