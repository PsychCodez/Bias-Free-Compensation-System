#!/usr/bin/env python3
"""
Test script for the Satellite Image Crop Analysis Web Application
"""

import os
import sys
import requests
import time
from pathlib import Path

def test_server_connection():
    """Test if the server is running"""
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("✓ Server is running and accessible")
            return True
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure it's running on localhost:5000")
        return False
    except Exception as e:
        print(f"✗ Error connecting to server: {e}")
        return False

def test_upload_endpoint():
    """Test the upload endpoint with a sample image"""
    try:
        # Create a simple test image
        from PIL import Image
        import numpy as np
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='green')
        test_image_path = 'test_image.jpg'
        test_image.save(test_image_path)
        
        # Test upload
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:5000/upload', files=files, timeout=30)
        
        # Clean up test image
        os.remove(test_image_path)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✓ Upload endpoint working correctly")
                print(f"  - Total fields detected: {data.get('total_fields', 0)}")
                return True
            else:
                print(f"✗ Upload failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ Upload endpoint returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing upload endpoint: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Testing Satellite Image Crop Analysis Web Application")
    print("=" * 50)
    
    # Test server connection
    if not test_server_connection():
        print("\nPlease start the server first by running: python run.py")
        sys.exit(1)
    
    print("\nTesting upload functionality...")
    if test_upload_endpoint():
        print("\n✓ All tests passed! The application is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the server logs for errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
