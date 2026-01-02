#!/usr/bin/env python3
"""
COMPLETE DEPLOYMENT SCRIPT FOR AI IMAGE COLORIZATION GAN
This script provides multiple deployment options: Web App, API, and Batch Processing
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
import threading
import requests

# ============================================================================
# MODEL ARCHITECTURES (Same as training/testing)
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
# INFERENCE UTILITIES
# ============================================================================

class ColorizationModel:
    def __init__(self, model_path='models/generator_final.pth', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Generator(input_channels=1, output_channels=3).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"❌ Model not found: {model_path}")
            print("Please train the model first using train_complete.py")
    
    def preprocess_image(self, image, target_size=64):
        """Preprocess image for colorization"""
        # Convert PIL Image to numpy if necessary
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            pass
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        
        # Resize image
        image = cv2.resize(image, (target_size, target_size))
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Normalize to [-1, 1] range
        gray_image = (gray_image.astype(np.float32) / 127.5) - 1.0
        
        # Convert to tensor
        gray_tensor = torch.from_numpy(gray_image).unsqueeze(0).unsqueeze(0)
        
        return gray_tensor, gray_image
    
    def colorize(self, image):
        """Colorize a single image"""
        # Preprocess
        gray_tensor, gray_np = self.preprocess_image(image)
        
        # Generate colorized image
        with torch.no_grad():
            gray_tensor = gray_tensor.to(self.device)
            colorized_tensor = self.model(gray_tensor)
            
            # Convert back to numpy
            colorized_image = colorized_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
            colorized_image = ((colorized_image + 1) * 127.5).astype(np.uint8)
        
        return colorized_image, gray_np
    
    def colorize_batch(self, images):
        """Colorize a batch of images"""
        results = []
        for image in images:
            colorized, gray = self.colorize(image)
            results.append({'colorized': colorized, 'grayscale': gray})
        return results

def process_image_for_display(image_array):
    """Process image array for Streamlit display"""
    # Handle different input formats
    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        # If values are in [-1, 1] range (normalized)
        if image_array.min() < 0:
            image_array = (image_array + 1) * 127.5
        # If values are in [0, 1] range
        elif image_array.max() <= 1.0:
            image_array = image_array * 255
    
    # Ensure proper range and type
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    return image_array

# Global model instance
colorization_model = None

def load_model():
    global colorization_model
    if colorization_model is None:
        colorization_model = ColorizationModel()
    return colorization_model

# ============================================================================
# STREAMLIT WEB APPLICATION
# ============================================================================

def run_streamlit_app():
    """Run Streamlit web application"""
    st.set_page_config(
        page_title="AI Image Colorization",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎨 AI Image Colorization with GANs")
    st.markdown("Transform your black and white images into vibrant colored photos using deep learning!")
    
    # Sidebar
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Quality Threshold", 0.0, 1.0, 0.5)
    show_processing_info = st.sidebar.checkbox("Show Processing Info", True)
    
    # Model loading
    if 'model' not in st.session_state:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_model()
    
    model = st.session_state.model
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a grayscale image to colorize"
        )
        
        # Sample images
        st.subheader("Or try a sample:")
        sample_choice = st.selectbox("Select a sample image", [
            "None", "Sample 1", "Sample 2", "Sample 3"
        ])
    
    with col2:
        st.header("🎨 Colorized Result")
        
        if uploaded_file is not None or sample_choice != "None":
            # Process uploaded image
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_container_width=True)
            else:
                # Handle sample images (you would need to provide sample images)
                st.info("Sample image functionality - add sample images to use this feature")
                image = None
            
            if image is not None:
                # Colorize button
                if st.button("🚀 Colorize Image", type="primary"):
                    with st.spinner("Colorizing image... This may take a moment."):
                        try:
                            colorized_image, grayscale_image = model.colorize(image)
            
                            # Process images for display
                            colorized_display = process_image_for_display(colorized_image)
                            grayscale_display = process_image_for_display(grayscale_image)
            
                            # Display results
                            col_result1, col_result2 = st.columns(2)
            
                            with col_result1:
                                st.subheader("Grayscale Input")
                                st.image(grayscale_display, caption="Grayscale Version")
            
                            with col_result2:
                                st.subheader("AI Colorized")
                                st.image(colorized_display, caption="Colorized Result")
                
                            # Success message
                            st.success("✅ Image processed successfully!")
            
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            # Add debug info
                            st.error(f"Debug info: Image shape: {image.size if hasattr(image, 'size') else 'unknown'}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This is a demo application. For best results, use clear grayscale images.")
    st.markdown("**Model Info:** Pix2Pix GAN trained on CIFAR-10 dataset")

# ============================================================================
# FLASK API SERVER
# ============================================================================

def create_flask_app():
    """Create Flask API application"""
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return jsonify({
            'message': 'AI Image Colorization API',
            'endpoints': {
                'colorize': '/colorize [POST]',
                'health': '/health [GET]',
                'info': '/info [GET]'
            }
        })
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'model_loaded': colorization_model is not None})
    
    @app.route('/info')
    def info():
        model = load_model()
        return jsonify({
            'model_type': 'Pix2Pix GAN',
            'input_size': '64x64',
            'device': str(model.device),
            'architecture': 'U-Net Generator + PatchGAN Discriminator'
        })
    
    @app.route('/colorize', methods=['POST'])
    def colorize():
        try:
            # Check if image file is present
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            
            # Load and process image
            image = Image.open(file.stream).convert('RGB')
            model = load_model()
            
            # Colorize
            colorized_image, grayscale_image = model.colorize(image)
            
            # Convert to base64 for JSON response
            pil_colorized = Image.fromarray(colorized_image)
            pil_grayscale = Image.fromarray((grayscale_image * 255).astype(np.uint8))
            
            # Encode images
            colorized_buffer = io.BytesIO()
            grayscale_buffer = io.BytesIO()
            
            pil_colorized.save(colorized_buffer, format='PNG')
            pil_grayscale.save(grayscale_buffer, format='PNG')
            
            colorized_base64 = base64.b64encode(colorized_buffer.getvalue()).decode()
            grayscale_base64 = base64.b64encode(grayscale_buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'colorized_image': colorized_base64,
                'grayscale_image': grayscale_base64,
                'format': 'PNG',
                'message': 'Image colorized successfully'
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_colorize_images(input_dir, output_dir, file_pattern='*.jpg'):
    """Batch process multiple images"""
    import glob
    from tqdm import tqdm
    
    # Load model
    model = load_model()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not image_files:
        print(f"No images found in {input_dir} with pattern {file_pattern}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    for image_path in tqdm(image_files, desc="Colorizing"):
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Colorize
            colorized_image, _ = model.colorize(image)
            
            # Save result
            output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_colorized.png'
            output_path = os.path.join(output_dir, output_filename)
            
            colorized_pil = Image.fromarray(colorized_image)
            colorized_pil.save(output_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"✓ Batch processing completed! Results saved to {output_dir}")

# ============================================================================
# DEPLOYMENT LAUNCHER
# ============================================================================

def main():
    """Main deployment launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Image Colorization Deployment")
    parser.add_argument('--mode', choices=['streamlit', 'api', 'batch'], 
                       default='streamlit', help='Deployment mode')
    parser.add_argument('--model-path', default='models/generator_final.pth',
                       help='Path to trained model')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port number for web services')
    parser.add_argument('--host', default='localhost',
                       help='Host address for web services')
    
    # Batch processing arguments
    parser.add_argument('--input-dir', help='Input directory for batch processing')
    parser.add_argument('--output-dir', help='Output directory for batch processing')
    parser.add_argument('--pattern', default='*.jpg', help='File pattern for batch processing')
    
    args = parser.parse_args()
    
    if args.mode == 'streamlit':
        print("🚀 Starting Streamlit Web App...")
        print(f"🌐 Access the app at: http://{args.host}:{args.port}")
        print("📝 Upload images and see them colorized in real-time!")
        
        # Note: In a real deployment, you would run this with streamlit CLI
        # streamlit run deploy_complete.py --server.port 8501
        run_streamlit_app()
        
    elif args.mode == 'api':
        print("🚀 Starting Flask API Server...")
        print(f"🌐 API available at: http://{args.host}:{args.port}")
        print("📖 API Endpoints:")
        print("   POST /colorize - Upload image for colorization")
        print("   GET  /health   - Check API health")
        print("   GET  /info     - Get model information")
        
        app = create_flask_app()
        app.run(host=args.host, port=args.port, debug=False)
        
    elif args.mode == 'batch':
        if not args.input_dir or not args.output_dir:
            print("❌ Batch mode requires --input-dir and --output-dir")
            return
        
        print("🚀 Starting Batch Processing...")
        print(f"📁 Input: {args.input_dir}")
        print(f"📁 Output: {args.output_dir}")
        print(f"🔍 Pattern: {args.pattern}")
        
        batch_colorize_images(args.input_dir, args.output_dir, args.pattern)

# ============================================================================
# SIMPLE INFERENCE SCRIPT
# ============================================================================

def simple_colorize(image_path, output_path=None):
    """Simple function to colorize a single image"""
    model = load_model()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"Loaded image: {image_path}")
    
    # Colorize
    print("Colorizing...")
    colorized_image, grayscale_image = model.colorize(image)
    
    # Save or display
    if output_path:
        colorized_pil = Image.fromarray(colorized_image)
        colorized_pil.save(output_path)
        print(f"✓ Colorized image saved to: {output_path}")
    else:
        # Display using matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(np.array(image))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(grayscale_image, cmap='gray')
        axes[1].set_title('Grayscale')
        axes[1].axis('off')
        
        axes[2].imshow(colorized_image)
        axes[2].set_title('AI Colorized')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return colorized_image

if __name__ == "__main__":
    main()

# Example usage:
# python deploy_complete.py --mode streamlit
# python deploy_complete.py --mode api --port 5000
# python deploy_complete.py --mode batch --input-dir ./test_images --output-dir ./results