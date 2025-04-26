# <center>UW-Madison GI Tract Image Segmentation</center>
![Alt cover_alt](images/cover.png)

## Overview
This project implements a deep learning solution for the [UW-Madison GI Tract Image Segmentation Kaggle Competition](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation). The goal is to segment the stomach and bowels in MRI scans to aid cancer treatment planning, using a dataset of 38,496 images from 85 patients provided by the UW-Madison Carbone Cancer Center.

### Sampled Case Scan Days:
<div style="display: flex; flex-direction: row; justify-content: center;">

|        Case: 2<br/>Day: 1         | Case: 15<br/>Day: 20                 |         Case: 101<br/>Day: 20          |  Case: 102<br/>Day: 0   |                 Case 156<br/>Day: 0                 |
|:---------------------------------:|--------------------------------------|:--------------------------------------:|:---:|:---------------------------------------------------:|
| ![Alt gif_2](images/gifs/case2_day1.gif) | ![Alt gif_15](images/gifs/case15_day20.gif) | ![Alt gif_101](images/gifs/case101_day20.gif) |  ![Alt gif_102](images/gifs/case102_day0.gif)   |        ![Alt gif_156](images/gifs/case156_day0.gif)        |

</div>

## Related Works
The proposed method builds on several key works in medical image segmentation, particularly U-Net-based architectures and efficient convolutional neural networks (CNN). Notable contributions include the original U-Net, its variants (UNet++, UNet 3+, nnU-Net), and lightweight encoders like MobileNetV2 and EfficientNet. See [References](#references) for details.

## Proposed Method
The proposed method utilizes a U-Net architecture with an **EfficientNet-B1** encoder, pre-trained on ImageNet, for semantic segmentation of the gastrointestinal tract in MRI scans. To adapt the single-channel (grayscale) input images from the UW-Madison dataset to the three-channel format expected by the pre-trained encoder, each image is converted to a three-channel representation by duplicating the grayscale channel across RGB. The model is fine-tuned on the competition dataset, consisting of 38,496 MRI images from 85 patients, to accurately segment the stomach, small bowel, and large bowel, optimizing for the competition’s multi-label Dice coefficient metric.

## Implementation

### Dataset
The dataset is sourced from the [UW-Madison GI Tract Image Segmentation Kaggle Competition](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data). It comprises 38,496 single-channel (grayscale) MRI scans from 85 patients, each with a resolution of 266x266 pixels, annotated for segmenting the **stomach**, **small bowel**, and **large bowel**. To ensure compatibility with model, images are converted to three-channel RGB format by duplicating the grayscale channel during preprocessing. Preprocessing also includes resizing images to 224x224 pixels and normalizing pixel values to [0, 1]. Data augmentation techniques, such as random horizontal flips, random crops, and photometric distortions, are applied to enhance model robustness.

>🔥 <span style="color: #dc7633">To accelerate training, all training and validation samples (~5GB) are loaded into memory, which is recommended for faster processing.</span>

#### Segmentation colors:
🔴 Large Bowel: Red

🟢Small Bowel: Green

🔵Stomach: Blue

### Model
The model is a U-Net architecture with an **EfficientNet-B1** encoder, pre-trained on ImageNet, implemented using the `segmentation_models_pytorch` library in PyTorch. The U-Net consists of an encoder-decoder structure with skip connections to preserve spatial details. The EfficientNet-B1 encoder extracts robust features, while the decoder upsamples these features to generate segmentation masks for three classes (stomach, small bowel, large bowel). The model outputs a multi-label segmentation map with shape (3, H, W), where each channel corresponds to a class.

<div style="display: flex; justify-content: center;"><img src="images/unet-process.png" alt="model_alt"/></div>

### Configurations

| Hyper-Parameter                                                                         |                                             | Values                                |
|-----------------------------------------------------------------------------------------|---------------------------------------------|---------------------------------------|
| <span style="color: orange">**Optimizer**<sub style="color: red">&nbsp;SGD</sub></span> | Learning Rate<br/>Weight Decay<br/>Momentum | 0.3<br/>1e-4<br/>0.9                  |
| <span style="color: orange">**Loss Function**</span>                                    | Dice Loss<br/>BCE Loss                      | %70<br/>     %30                      |
| <span style="color: orange">**Batch Size**</span>                                       | Train Set<br/>Eval and Test Sets            | 64<br/>128                            |
| <span style="color: orange">**Input Size**</span>                                       |                                             | 224 x 224                             |
| <span style="color: orange">**Epochs**</span>                                           |                                             | 8                                     |
| <span style="color: orange">**Device**</span>                                           |                                             | Google Colab (NVIDIA T4)              |
| <span style="color: orange">**Splitting**</span>                                        |                                             | train.txt - validation.txt - test.txt |
| <span style="color: orange">**Backbone**<sub>U-Net Encoder</sub></span>                 | Efficient-net 1B                            | 8,757,39 parameters                   |


### Train
The training pipeline includes:
1. Loading the preprocessed three-channel images and corresponding segmentation masks.
2. Applying data augmentations (random horizontal flips, random crops, and photometric distortions) using the `torchvision.transforms.v2` module.
3. Forward and backward passes through the U-Net model, optimizing the combined Dice+BCE loss.
4. Monitoring validation loss and saving the best model checkpoint based on the validation Dice coefficient.
Training is performed on a GPU to accelerate computation, with a batch size of 16 and a maximum of 8 epochs. The EfficientNet-B1 encoder is initialized with ImageNet weights, and all layers are fine-tuned to adapt to the medical imaging domain.

### Evaluate
The model is evaluated using the competition’s primary metric: the **multi-label Dice coefficient**, which measures segmentation accuracy for the stomach, small bowel, and large bowel. The Dice coefficient is computed per class and averaged across all images in the test set. The evaluation pipeline involves:
1. Loading the test dataset and preprocessing images (three-channel conversion, resizing, normalization, ...).
2. Generating segmentation masks using the trained model in inference mode.
3. Computing the Dice coefficient for each class and averaging to obtain the final score.
The model achieves a competitive Dice score on the evaluation set. Results are analyzed to identify segmentation challenges, such as small bowel boundaries, for future improvements.

## Results
### Losses and Scores:

|   Phase    | Loss (Dice loss + BCE loss) | Dice Score |
|:----------:|:---------------------------:|:----------:|
|   Train    |            0.931            |   0.873    |
| Evaluation |            0.168            |   0.781    |
|    Test    |            0.164            |   0.787    |


### Learning and Metric Curves
<div><img src="./images/learning_curve.PNG" alt="curves_alt"/></div>

### Segments visualization
<div style="display: flex; justify-content: center;"><img src="images/figure.png" alt="result_alt" width="900"/></div>

## Requirements
Use requirements.txt as required dependencies:

| Package                                        | Version                                |
|------------------------------------------------|----------------------------------------|
| torch                                          | 2.6.0+cu124                            |
| torchvision                                    | 0.21.0+cu124                           |
| ⚠️<span style="color: red">torchmetrics</span> | <span style="color: red">1.2.0</span>  |
| segmentation_models_pytorch                    | 0.5.0                                  |
| tqdm                                           | 4.67.1                                 |

## Contact
For questions or collaboration, reach out to:
- **Email**: [yassingourkani@outlook.com](mailto:yassingourkani@outlook.com)
- **LinkedIn**: [Yassin Gourkani](https://www.linkedin.com/in/yassingourkani/)

## References
| Author(s) | Year | Title | Key Contribution | Reference |
|-----------|------|-------|------------------|-----------|
| Ronneberger et al. | 2015 | U-Net: Convolutional Networks for Biomedical Image Segmentation | Introduced U-Net, a foundational CNN architecture for medical image segmentation, widely used for segmenting organs like the GI tract. | Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015*, 234-241. |
| Zhou et al. | 2018 | UNet++: A Nested U-Net Architecture for Medical Image Segmentation | Proposed UNet++, enhancing U-Net with nested skip connections for better feature aggregation, improving segmentation accuracy. | Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A nested U-Net architecture for medical image segmentation. *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 3-11. |
| Sandler et al. | 2018 | MobileNetV2: Inverted Residuals and Linear Bottlenecks | Developed MobileNetV2, a lightweight CNN used as an encoder in segmentation models like UMobileNetV2 for efficient GI tract segmentation. | Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4510-4520. |
| Isensee et al. | 2018 | nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation | Introduced nnU-Net, a self-configuring U-Net framework that optimizes for diverse medical segmentation tasks, improving robustness. | Isensee, F., Petersen, J., Klein, A., et al. (2018). nnU-Net: Self-adapting framework for U-Net-based medical image segmentation. *arXiv preprint arXiv:1809.10486*. |
| Tan & Le | 2019 | EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | Proposed EfficientNet, a scalable CNN architecture used as the encoder in the proposed method. | Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning*, 6105-6114. |
| Huang et al. | 2020 | UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation | Introduced UNet 3+, combining multi-scale features for improved segmentation of complex structures like the GI tract. | Huang, H., Lin, L., Tong, R., et al. (2020). UNet 3+: A full-scale connected UNet for medical image segmentation. *ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing*, 1055-1059. |
| Jha et al. | 2023 | UMobileNetV2 Model for Semantic Segmentation of Gastrointestinal Tract in MRI Scans | Proposed UMobileNetV2, integrating MobileNetV2 as the encoder in a U-Net architecture, evaluated on the UW-Madison dataset for GI tract segmentation. | Jha, D., Singh, S., & Gupta, S. (2023). UMobileNetV2 model for semantic segmentation of gastrointestinal tract in MRI scans. *Journal of Medical Imaging*, 10(4), 044002. |