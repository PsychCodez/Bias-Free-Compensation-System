#!/usr/bin/env python3
"""
Satellite Image Crop Analysis Web Application
Startup script for the Flask application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import flask
        import tensorflow
        import cv2
        import PIL
        import numpy
        import matplotlib
        import pandas
        import segmentation_models_pytorch
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_model_files():
    """Check if model files exist"""
    model_path = Path("Boundary/ParcelDelineation/best-unet-sentinel.hdf5")
    if model_path.exists():
        print("✓ Model file found")
        return True
    else:
        print("✗ Model file not found at:", model_path)
        print("Please ensure the trained model is in the correct location")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'results', 'templates']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✓ Directories created")

def main():
    """Main startup function"""
    print("=" * 50)
    print("Satellite Image Crop Analysis Web Application")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        print("Warning: Model file not found. The application may not work properly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\nStarting Flask application...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask application
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
