#!/usr/bin/env python3
"""
COMPLETE TRAINING SCRIPT FOR AI IMAGE COLORIZATION GAN
This script handles the entire training process from start to finish.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import time
from datetime import datetime

# ============================================================================
# MODEL ARCHITECTURES
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
        
        # Encoder (Downsampling) - Reduced to 6 layers for 64x64 input
        self.enc1 = ConvBlock(input_channels, 64, use_batchnorm=False)  # 64->32
        self.enc2 = ConvBlock(64, 128)     # 32->16
        self.enc3 = ConvBlock(128, 256)    # 16->8
        self.enc4 = ConvBlock(256, 512)    # 8->4
        self.enc5 = ConvBlock(512, 512)    # 4->2
        self.enc6 = ConvBlock(512, 512)    # 2->1
        
        # Decoder (Upsampling) - Corresponding 6 layers
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


class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    def __init__(self, input_channels=1, target_channels=3):
        super(Discriminator, self).__init__()
        
        in_channels = input_channels + target_channels
        
        self.conv1 = ConvBlock(in_channels, 64, use_batchnorm=False)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512, stride=1)
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, grayscale, color):
        x = torch.cat([grayscale, color], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)
        return x

# ============================================================================
# DATASET HANDLING
# ============================================================================

class CIFAR10ColorizationDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        color_image, _ = self.original_dataset[idx]
        
        # Convert to grayscale
        gray_image = torch.mean(color_image, dim=0, keepdim=True)
        
        return gray_image, color_image

def create_dataloaders(batch_size=16, image_size=64):
    """Create CIFAR-10 dataloaders for colorization"""
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Convert to colorization datasets
    train_colorization = CIFAR10ColorizationDataset(train_dataset)
    test_colorization = CIFAR10ColorizationDataset(test_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(train_colorization, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_colorization, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class GANLoss:
    def __init__(self, device, lambda_l1=100):
        self.device = device
        self.lambda_l1 = lambda_l1
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.bce_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output))
        total_loss = (real_loss + fake_loss) * 0.5
        return total_loss
    
    def generator_loss(self, fake_output, fake_images, real_images):
        adversarial_loss = self.bce_loss(fake_output, torch.ones_like(fake_output))
        l1_loss = self.l1_loss(fake_images, real_images)
        total_loss = adversarial_loss + (self.lambda_l1 * l1_loss)
        return total_loss, adversarial_loss, l1_loss

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def save_sample_images(generator, dataloader, device, epoch, save_dir='training_progress', num_samples=4):
    """Save sample colorized images during training"""
    generator.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        grayscale_batch, color_batch = next(iter(dataloader))
        grayscale_batch = grayscale_batch[:num_samples].to(device)
        color_batch = color_batch[:num_samples].to(device)
        
        generated_batch = generator(grayscale_batch)
        
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
        
        for i in range(num_samples):
            # Denormalize for display
            gray_img = (grayscale_batch[i].cpu().squeeze() + 1) * 127.5
            real_img = ((color_batch[i].cpu() + 1) * 127.5).byte().numpy().transpose(1, 2, 0)
            gen_img = ((generated_batch[i].cpu() + 1) * 127.5).byte().numpy().transpose(1, 2, 0)
            
            axes[0, i].imshow(gray_img, cmap='gray')
            axes[0, i].set_title('Input')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(real_img)
            axes[1, i].set_title('Target')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(gen_img)
            axes[2, i].set_title('Generated')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    generator.train()

def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, 
                   g_losses, d_losses, checkpoint_dir='checkpoints'):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_gan():
    """Main training function"""
    
    # Training configuration
    config = {
        'num_epochs': 50,
        'batch_size': 16,
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'lambda_l1': 100,
        'image_size': 64,
        'save_every': 10,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print("=" * 60)
    print("AI IMAGE COLORIZATION GAN - TRAINING")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Image size: {config['image_size']}x{config['image_size']}")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('training_progress', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Create datasets
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = create_dataloaders(config['batch_size'], config['image_size'])
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    
    # Create models
    print("Creating models...")
    generator = Generator(input_channels=1, output_channels=3).to(config['device'])
    discriminator = Discriminator(input_channels=1, target_channels=3).to(config['device'])
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Create optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], 0.999))
    
    # Loss function
    criterion = GANLoss(config['device'], config['lambda_l1'])
    
    # Training history
    g_losses = []
    d_losses = []
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for batch_idx, (grayscale_images, color_images) in enumerate(progress_bar):
            grayscale_images = grayscale_images.to(config['device'])
            color_images = color_images.to(config['device'])
            
            # Train Discriminator
            optimizer_d.zero_grad()
            fake_images = generator(grayscale_images)
            real_output = discriminator(grayscale_images, color_images)
            fake_output = discriminator(grayscale_images, fake_images.detach())
            d_loss = criterion.discriminator_loss(real_output, fake_output)
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            fake_output = discriminator(grayscale_images, fake_images)
            g_loss, adv_loss, l1_loss = criterion.generator_loss(fake_output, fake_images, color_images)
            g_loss.backward()
            optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}',
                'L1': f'{l1_loss.item():.4f}'
            })
        
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}] - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}')
        
        # Save sample images and checkpoints
        if (epoch + 1) % config['save_every'] == 0:
            save_sample_images(generator, test_loader, config['device'], epoch + 1)
            save_checkpoint(generator, discriminator, optimizer_g, optimizer_d,
                           epoch + 1, g_losses, d_losses)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/3600:.2f} hours")
    
    # Save final models
    torch.save(generator.state_dict(), 'models/generator_final.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator_final.pth')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss', color='blue')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(d_losses, label='Discriminator Loss', color='red')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Training completed successfully!")
    print("✓ Models saved to models/")
    print("✓ Training progress saved to training_progress/")
    print("✓ Checkpoints saved to checkpoints/")
    
    return generator, discriminator, g_losses, d_losses

if __name__ == "__main__":
    # Run training
    generator, discriminator, g_losses, d_losses = train_gan()