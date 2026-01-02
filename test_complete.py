#!/usr/bin/env python3
"""
COMPLETE TESTING & EVALUATION SCRIPT FOR AI IMAGE COLORIZATION GAN
This script handles model evaluation, testing, and performance analysis.
"""

import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import glob
from tqdm import tqdm
import seaborn as sns

# ============================================================================
# MODEL ARCHITECTURES (Same as training)
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, 
                 padding=1, use_batchnorm=True, use_activation=True, activation='leaky'):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.use_batchnorm = use_batchnorm
        self.use_activation = use_activation
        
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
            
        if use_activation:
            if activation == 'leaky':
                self.activation = nn.LeakyReLU(0.2, inplace=True)
            elif activation == 'relu':
                self.activation = nn.ReLU(inplace=True)
            elif activation == 'tanh':
                self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        if self.use_activation:
            x = self.activation(x)
        return x

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, 
                 padding=1, use_batchnorm=True, use_dropout=False, activation='relu'):
        super(ConvTransposeBlock, self).__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, 
                                               kernel_size, stride, padding, bias=False)
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
            
        if use_dropout:
            self.dropout = nn.Dropout2d(0.5)
            
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.conv_transpose(x)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class Generator(nn.Module):
    """U-Net Generator optimized for 64x64 images (CIFAR-10)"""
    def __init__(self, input_channels=1, output_channels=3):
        super(Generator, self).__init__()
        
        # Encoder (Downsampling) - 6 layers for 64x64 input
        self.enc1 = ConvBlock(input_channels, 64, use_batchnorm=False)  # 64->32
        self.enc2 = ConvBlock(64, 128)     # 32->16
        self.enc3 = ConvBlock(128, 256)    # 16->8
        self.enc4 = ConvBlock(256, 512)    # 8->4
        self.enc5 = ConvBlock(512, 512)    # 4->2
        self.enc6 = ConvBlock(512, 512)    # 2->1
        
        # Decoder (Upsampling) - 6 layers
        self.dec1 = ConvTransposeBlock(512, 512, use_dropout=True)      # 1->2
        self.dec2 = ConvTransposeBlock(1024, 512, use_dropout=True)     # 2->4
        self.dec3 = ConvTransposeBlock(1024, 256)                       # 4->8
        self.dec4 = ConvTransposeBlock(512, 128)                        # 8->16
        self.dec5 = ConvTransposeBlock(256, 64)                         # 16->32
        
        # Final layer
        self.final = nn.ConvTranspose2d(128, output_channels, 4, 2, 1)  # 32->64
        self.final_activation = nn.Tanh()
        
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)       # 64 -> 32
        e2 = self.enc2(e1)      # 32 -> 16
        e3 = self.enc3(e2)      # 16 -> 8
        e4 = self.enc4(e3)      # 8 -> 4
        e5 = self.enc5(e4)      # 4 -> 2
        e6 = self.enc6(e5)      # 2 -> 1
        
        # Decoder with skip connections
        d1 = self.dec1(e6)                              # 1 -> 2
        d1 = torch.cat([d1, e5], dim=1)                 # Skip connection
        
        d2 = self.dec2(d1)                              # 2 -> 4
        d2 = torch.cat([d2, e4], dim=1)                 # Skip connection
        
        d3 = self.dec3(d2)                              # 4 -> 8
        d3 = torch.cat([d3, e3], dim=1)                 # Skip connection
        
        d4 = self.dec4(d3)                              # 8 -> 16
        d4 = torch.cat([d4, e2], dim=1)                 # Skip connection
        
        d5 = self.dec5(d4)                              # 16 -> 32
        d5 = torch.cat([d5, e1], dim=1)                 # Skip connection
        
        # Final output
        output = self.final(d5)                         # 32 -> 64
        output = self.final_activation(output)
        
        return output


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def load_trained_generator(model_path, device):
    """Load the trained generator model"""
    generator = Generator(input_channels=1, output_channels=3).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator

def preprocess_image(image_path, target_size=64):
    """Preprocess a single image for colorization"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, (target_size, target_size))
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalize to [-1, 1] range
    gray_image = (gray_image.astype(np.float32) / 127.5) - 1.0
    color_image = (image.astype(np.float32) / 127.5) - 1.0
    
    # Convert to tensors
    gray_tensor = torch.from_numpy(gray_image).unsqueeze(0).unsqueeze(0)
    color_tensor = torch.from_numpy(color_image.transpose(2, 0, 1)).unsqueeze(0)
    
    return gray_tensor, color_tensor, gray_image, color_image

def colorize_single_image(generator, gray_tensor, device):
    """Colorize a single grayscale image"""
    generator.eval()
    with torch.no_grad():
        gray_tensor = gray_tensor.to(device)
        colorized_tensor = generator(gray_tensor)
        
        # Convert back to numpy
        colorized_image = colorized_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
        colorized_image = ((colorized_image + 1) * 127.5).astype(np.uint8)
        
    return colorized_image

def calculate_metrics(real_image, generated_image):
    """Calculate PSNR and SSIM metrics"""
    # Ensure images are in the right format
    if real_image.max() <= 1.0:
        real_image = (real_image * 255).astype(np.uint8)
    if generated_image.max() <= 1.0:
        generated_image = (generated_image * 255).astype(np.uint8)
    
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(real_image, generated_image, data_range=255)
    
    # Calculate SSIM
    if len(real_image.shape) == 3:
        ssim = structural_similarity(real_image, generated_image, 
                                   multichannel=True, data_range=255, channel_axis=2)
    else:
        ssim = structural_similarity(real_image, generated_image, data_range=255)
    
    return psnr, ssim

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_on_test_set(generator, test_images_dir, device, num_samples=100):
    """Evaluate model on test images"""
    print("=" * 60)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 60)
    
    # Get test image paths
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    if not image_paths:
        print(f"No images found in {test_images_dir}")
        return None
    
    # Limit number of samples
    if num_samples and len(image_paths) > num_samples:
        image_paths = np.random.choice(image_paths, num_samples, replace=False)
    
    print(f"Evaluating on {len(image_paths)} images...")
    
    psnr_scores = []
    ssim_scores = []
    results = []
    
    for i, image_path in enumerate(tqdm(image_paths, desc="Evaluating")):
        try:
            # Preprocess image
            gray_tensor, color_tensor, gray_np, color_np = preprocess_image(image_path)
            
            # Colorize
            colorized_image = colorize_single_image(generator, gray_tensor, device)
            
            # Calculate metrics
            color_np_uint8 = ((color_np + 1) * 127.5).astype(np.uint8)
            psnr, ssim = calculate_metrics(color_np_uint8, colorized_image)
            
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            
            results.append({
                'image_path': image_path,
                'psnr': psnr,
                'ssim': ssim,
                'gray_image': gray_np,
                'real_image': color_np_uint8,
                'generated_image': colorized_image
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Calculate statistics
    metrics = {
        'mean_psnr': np.mean(psnr_scores),
        'std_psnr': np.std(psnr_scores),
        'mean_ssim': np.mean(ssim_scores),
        'std_ssim': np.std(ssim_scores),
        'min_psnr': np.min(psnr_scores),
        'max_psnr': np.max(psnr_scores),
        'min_ssim': np.min(ssim_scores),
        'max_ssim': np.max(ssim_scores),
        'num_images': len(psnr_scores)
    }
    
    return metrics, results, psnr_scores, ssim_scores

def test_on_cifar10(generator, device, num_samples=100):
    """Test on CIFAR-10 test set"""
    print("=" * 60)
    print("TESTING ON CIFAR-10 DATASET")
    print("=" * 60)
    
    # Load CIFAR-10 test set
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
        batch_size=1, shuffle=True
    )
    
    psnr_scores = []
    ssim_scores = []
    results = []
    
    print(f"Testing on {num_samples} CIFAR-10 images...")
    
    generator.eval()
    with torch.no_grad():
        for i, (color_image, _) in enumerate(tqdm(test_dataset, total=num_samples, desc="Testing")):
            if i >= num_samples:
                break
            
            # Convert to grayscale
            gray_image = torch.mean(color_image, dim=1, keepdim=True)
            
            # Generate colorized image
            gray_image = gray_image.to(device)
            color_image = color_image.to(device)
            
            generated_image = generator(gray_image)
            
            # Convert to numpy for metrics
            gray_np = gray_image.cpu().squeeze().numpy()
            real_np = ((color_image.cpu().squeeze() + 1) * 127.5).byte().numpy().transpose(1, 2, 0)
            gen_np = ((generated_image.cpu().squeeze() + 1) * 127.5).byte().numpy().transpose(1, 2, 0)
            
            # Calculate metrics
            psnr, ssim = calculate_metrics(real_np, gen_np)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            
            results.append({
                'gray_image': gray_np,
                'real_image': real_np,
                'generated_image': gen_np,
                'psnr': psnr,
                'ssim': ssim
            })
    
    # Calculate statistics
    metrics = {
        'mean_psnr': np.mean(psnr_scores),
        'std_psnr': np.std(psnr_scores),
        'mean_ssim': np.mean(ssim_scores),
        'std_ssim': np.std(ssim_scores),
        'min_psnr': np.min(psnr_scores),
        'max_psnr': np.max(psnr_scores),
        'min_ssim': np.min(ssim_scores),
        'max_ssim': np.max(ssim_scores),
        'num_images': len(psnr_scores)
    }
    
    return metrics, results, psnr_scores, ssim_scores

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_results(results, num_samples=8, save_path='test_results.png'):
    """Visualize test results"""
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    
    for i in range(min(num_samples, len(results))):
        result = results[i]
        
        # Grayscale input
        axes[0, i].imshow(result['gray_image'], cmap='gray')
        axes[0, i].set_title('Input')
        axes[0, i].axis('off')
        
        # Real color image
        axes[1, i].imshow(result['real_image'])
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        # Generated color image
        axes[2, i].imshow(result['generated_image'])
        axes[2, i].set_title(f'Generated\nPSNR: {result["psnr"]:.1f}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_metrics_distribution(psnr_scores, ssim_scores, save_path='metrics_distribution.png'):
    """Plot distribution of evaluation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR distribution
    axes[0].hist(psnr_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(psnr_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(psnr_scores):.2f}')
    axes[0].set_title('PSNR Distribution')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM distribution
    axes[1].hist(ssim_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(np.mean(ssim_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(ssim_scores):.3f}')
    axes[1].set_title('SSIM Distribution')
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def generate_evaluation_report(metrics, save_path='evaluation_report.txt'):
    """Generate detailed evaluation report"""
    with open(save_path, 'w',encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("AI IMAGE COLORIZATION GAN - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"DATASET STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of test images: {metrics['num_images']}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"PSNR (Peak Signal-to-Noise Ratio):\n")
        f.write(f"  Mean: {metrics['mean_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB\n")
        f.write(f"  Range: {metrics['min_psnr']:.2f} - {metrics['max_psnr']:.2f} dB\n\n")
        
        f.write(f"SSIM (Structural Similarity Index):\n")
        f.write(f"  Mean: {metrics['mean_ssim']:.3f} ± {metrics['std_ssim']:.3f}\n")
        f.write(f"  Range: {metrics['min_ssim']:.3f} - {metrics['max_ssim']:.3f}\n\n")
        
        f.write("PERFORMANCE INTERPRETATION:\n")
        f.write("-" * 30 + "\n")
        
        if metrics['mean_psnr'] > 25:
            f.write("✓ PSNR: Excellent reconstruction quality\n")
        elif metrics['mean_psnr'] > 20:
            f.write("✓ PSNR: Good reconstruction quality\n")
        elif metrics['mean_psnr'] > 15:
            f.write("○ PSNR: Fair reconstruction quality\n")
        else:
            f.write("✗ PSNR: Poor reconstruction quality\n")
        
        if metrics['mean_ssim'] > 0.8:
            f.write("✓ SSIM: Excellent structural similarity\n")
        elif metrics['mean_ssim'] > 0.6:
            f.write("✓ SSIM: Good structural similarity\n")
        elif metrics['mean_ssim'] > 0.4:
            f.write("○ SSIM: Fair structural similarity\n")
        else:
            f.write("✗ SSIM: Poor structural similarity\n")
    
    print(f"Evaluation report saved to: {save_path}")

# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def run_complete_evaluation():
    """Run complete model evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    model_path = 'models/generator_final.pth'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first using train_complete.py")
        return
    
    print("Loading trained generator...")
    generator = load_trained_generator(model_path, device)
    
    # Test on CIFAR-10
    print("\nTesting on CIFAR-10...")
    metrics, results, psnr_scores, ssim_scores = test_on_cifar10(generator, device, num_samples=500)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean PSNR: {metrics['mean_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB")
    print(f"Mean SSIM: {metrics['mean_ssim']:.3f} ± {metrics['std_ssim']:.3f}")
    print(f"PSNR Range: [{metrics['min_psnr']:.2f}, {metrics['max_psnr']:.2f}] dB")
    print(f"SSIM Range: [{metrics['min_ssim']:.3f}, {metrics['max_ssim']:.3f}]")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results, num_samples=8, save_path='evaluation_results/test_samples.png')
    plot_metrics_distribution(psnr_scores, ssim_scores, save_path='evaluation_results/metrics_distribution.png')
    
    # Generate report
    generate_evaluation_report(metrics, save_path='evaluation_results/evaluation_report.txt')
    
    print("✓ Evaluation completed!")
    print("✓ Results saved to evaluation_results/")
    
    return metrics, results

def test_single_image(image_path, output_path=None):
    """Test the model on a single image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = 'models/generator_final.pth'
    generator = load_trained_generator(model_path, device)
    
    # Process image
    gray_tensor, color_tensor, gray_np, color_np = preprocess_image(image_path)
    colorized_image = colorize_single_image(generator, gray_tensor, device)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gray_np, cmap='gray')
    axes[0].set_title('Grayscale Input')
    axes[0].axis('off')
    
    axes[1].imshow(((color_np + 1) * 127.5).astype(np.uint8))
    axes[1].set_title('Original Color')
    axes[1].axis('off')
    
    axes[2].imshow(colorized_image)
    axes[2].set_title('AI Colorized')
    axes[2].axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to: {output_path}")
    
    plt.show()
    
    return colorized_image

if __name__ == "__main__":
    # Run complete evaluation
    metrics, results = run_complete_evaluation()
    
    # Optional: Test on a single image
    # test_single_image('path/to/your/image.jpg', 'colorized_result.png')