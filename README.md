# AI Image Colorization using Pix2Pix GAN

This project implements an automatic **image colorization** system that converts grayscale images into realistic RGB images using a **Pix2Pix conditional GAN** with a **U-Net generator** and **PatchGAN discriminator**.

---

## Overview

The goal of this project is to learn a mapping from **grayscale images** to their **corresponding color images** in a **supervised, paired** setting. A conditional GAN framework is used, where:

- The **generator** learns to colorize grayscale images.
- The **discriminator** learns to distinguish real color images from generated ones, given the same grayscale input.

The model is trained and evaluated on the **CIFAR-10** dataset, which provides diverse natural images across 10 classes.

---

## Model Architecture

### Generator: U-Net

The generator follows a **U-Net encoder–decoder** architecture:

- **Encoder (Downsampling path)**  
  - Series of convolutional layers with kernel size 4, stride 2, and padding 1.  
  - Uses **LeakyReLU** activations and **batch normalization** (except the first layer).  
  - Progressively reduces spatial resolution while increasing feature depth to capture global and local context.

- **Decoder (Upsampling path)**  
  - Transposed convolution (ConvTranspose2d) layers mirror the encoder, doubling spatial size at each step.  
  - Uses **ReLU** activations, **batch normalization**, and **dropout** in early decoder layers to improve generalization.  
  - **Skip connections** concatenate encoder feature maps with decoder layers at corresponding resolutions, preserving edges and fine spatial details critical for accurate color placement.

- **Output Layer**  
  - Final transposed convolution maps features to a 3-channel RGB image.  
  - Uses **Tanh** activation so outputs are in the range [-1, 1], matching normalized training data.

### Discriminator: PatchGAN

The discriminator is a **PatchGAN** CNN that classifies local image patches as real or fake rather than the whole image.

- Input: concatenation of **grayscale image** and **color image** (either real or generated) along the channel dimension.  
- Architecture: about **4 convolutional layers** with kernel size 4, stride 2, and LeakyReLU activations, followed by a final convolution that outputs an N×N map of real/fake probabilities (one per patch).  
- This design focuses on **high-frequency, local realism** (textures, edges, small details), which is crucial for natural-looking colorization.

---

## Loss Functions

### Generator Loss

The generator minimizes a **combined loss**:

1. **Adversarial Loss**  
   - Binary cross-entropy loss encouraging the discriminator to classify generated (fake) images as real.  
   - Drives the generator to produce **visually realistic** colorizations.

2. **L1 Reconstruction Loss**  
   - Mean absolute error between generated color image and ground truth color image.  
   - Encourages **pixel-level accuracy** and helps preserve structure and content.  
   - L1 is preferred over L2 to reduce blurring and produce sharper outputs.

The final generator loss is:
L_G = L_adv + λ · L_1
where λ = 100 balances color accuracy with realism.

### Discriminator Loss

The discriminator uses a standard **binary cross-entropy** objective:

- Maximizes the probability of assigning label **real (1)** to real grayscale–color pairs.  
- Maximizes the probability of assigning label **fake (0)** to grayscale–generated pairs.

This adversarial game between generator and discriminator leads to progressively more realistic colorizations.

---

## Training Setup

- **Dataset:** CIFAR-10 (60,000 natural images across 10 classes). Color images are converted to grayscale for input, with original RGB used as ground truth targets.  
- **Input/Output Resolution:** Images are resized to **64×64** for training and inference.  
- **Normalization:** Both grayscale and color images are normalized to [-1, 1] to match Tanh outputs.

### Hyperparameters

- **Optimizer:** Adam for both generator and discriminator.  
  - Learning Rate: **0.0002** – low enough to stabilize adversarial training and prevent generator–discriminator imbalance.  
  - β₁ = 0.5 – reduced momentum, commonly used in GANs for more stable updates.
- **Batch Size:** **16** – chosen to fit within a 6 GB VRAM GPU (e.g., GTX 1660 Ti) while maintaining stable gradient estimates.  
- **Epochs:** **50** – sufficient for convergence on CIFAR-10 scale without severe overfitting.  
- **Lambda L1:** **100** – ensures that reconstruction accuracy is strongly emphasized alongside adversarial realism.

---

## Evaluation

The model is evaluated using:

- **PSNR (Peak Signal-to-Noise Ratio)**  
  - Measures the pixel-wise difference (via MSE) between generated and ground truth images.  
  - Higher PSNR indicates closer numerical reconstruction quality.

- **SSIM (Structural Similarity Index Measure)**  
  - Assesses similarity in luminance, contrast, and structure between generated and real images.  
  - Values close to 1 indicate high perceptual similarity and good structural preservation.

### Results

- **PSNR:** approximately **24.3 dB**  
- **SSIM:** approximately **0.898**

These values are competitive with many CNN- and GAN-based colorization methods reported in the literature for automatic colorization on moderate-resolution datasets.

---

## Related Work

This project is directly inspired by and aligned with:

- **Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks** (Isola et al., CVPR 2017) – introduces the U-Net generator and PatchGAN discriminator architecture used here, along with the combined adversarial + L1 loss for paired image translation tasks.  
- **Colorful Image Colorization** (Zhang et al., ECCV 2016) – demonstrates the effectiveness of deep CNNs for automatic colorization.  
- Survey work on grayscale image colorization highlighting GAN-based approaches as a strong direction for fully automatic, realistic colorization of natural images.

---

## Key References

1. Isola, P., Zhu, J.Y., Zhou, T., & Efros, A.A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017.
2. Zhang, R., Isola, P., & Efros, A.A. (2016). Colorful Image Colorization. ECCV 2016.
3. Žeger, I., Grgić, S., Vuković, J., & Šišul, G. (2021). Grayscale Image Colorization Methods: Overview and Evaluation. IEEE Access.
4. Goodfellow, I., et al. (2014). Generative Adversarial Nets. NeurIPS 2014.

---

## Technologies Used

- **PyTorch** – Deep learning framework for model implementation
- **CIFAR-10 Dataset** – Training and evaluation data
- **Python** – Primary programming language
- **CUDA** – GPU acceleration (GTX 1660 Ti, 6GB VRAM)

---

## Future Improvements

- Train on higher resolution images (128×128, 256×256)
- Experiment with perceptual loss (VGG-based) for improved visual quality
- Integrate attention mechanisms for better global context
- Use LAB color space instead of RGB for more natural colorization
- Apply the model to historical photo restoration tasks

---
