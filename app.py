from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the Boundary/ParcelDelineation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Boundary', 'ParcelDelineation'))

from utils.data_loader_utils import read_imgs_keraspp
from models.unet import unet
import segmentation_models_pytorch as sm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables for models
parcel_model = None
BACKBONE = 'resnet34'
preprocess_input = None

def ensure_model_loaded():
    """Ensure the model and preprocessing function are loaded"""
    global parcel_model, preprocess_input
    if preprocess_input is None or parcel_model is None:
        load_parcel_model()

def load_parcel_model():
    """Load the parcel delineation model"""
    global parcel_model, preprocess_input
    try:
        # Initialize preprocessing function
        preprocess_input = sm.encoders.get_preprocessing_fn(BACKBONE, pretrained='imagenet')
        # Load the trained U-Net model
        model_path = os.path.join('Boundary', 'ParcelDelineation', 'best-unet-sentinel.hdf5')
        if os.path.exists(model_path):
            # Load custom dependencies
            dependencies = {
                'f1': f1_score,
                'dice_coef_sim': dice_coef_sim
            }
            parcel_model = load_model(model_path, custom_objects=dependencies)
            print("Parcel delineation model loaded successfully")
        else:
            print(f"Model file not found at {model_path}")
    except Exception as e:
        print(f"Error loading parcel model: {e}")

def f1_score(y_true, y_pred):
    """F1 score metric for model loading"""
    from tensorflow.keras import backend as K
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2*((precision_val*recall_val)/(precision_val+recall_val+K.epsilon()))

def dice_coef_sim(y_true, y_pred):
    """Dice coefficient metric for model loading"""
    from tensorflow.keras import backend as K
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input - creates 9-channel stacked input"""
    try:
        # Ensure model and preprocessing function are loaded
        ensure_model_loaded()
        
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        
        # Normalize to 0-1 range
        img_array = img_array.astype(np.float32) / 255.0
        
        # Apply preprocessing to the 3-channel image first
        img_array_batch = np.expand_dims(img_array, axis=0)
        print(f"Before preprocessing shape: {img_array_batch.shape}")
        print(f"Before preprocessing min/max: {img_array_batch.min():.4f}/{img_array_batch.max():.4f}")
        
        preprocessed_img = preprocess_input(img_array_batch)
        print(f"After preprocessing shape: {preprocessed_img.shape}")
        print(f"After preprocessing min/max: {preprocessed_img.min():.4f}/{preprocessed_img.max():.4f}")
        
        # Create 9-channel stacked input (replicate the preprocessed image 3 times)
        # This simulates the prior, central, and anterior images from the training data
        stacked_array = np.zeros((1, target_size[0], target_size[1], 9))
        stacked_array[:, :, :, 0:3] = preprocessed_img  # Prior image (channels 0-2)
        stacked_array[:, :, :, 3:6] = preprocessed_img  # Central image (channels 3-5)
        stacked_array[:, :, :, 6:9] = preprocessed_img  # Anterior image (channels 6-8)
        
        print(f"Final stacked array shape: {stacked_array.shape}")
        print(f"Final stacked array min/max: {stacked_array.min():.4f}/{stacked_array.max():.4f}")
        
        return stacked_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_parcel_delineation(image_array):
    """Predict parcel boundaries using the loaded model"""
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        if parcel_model is None:
            print("Model is None, cannot make prediction")
            return None
        
        print(f"Input shape: {image_array.shape}")
        
        # Make prediction
        prediction = parcel_model.predict(image_array, verbose=1)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction min/max: {prediction.min():.4f}/{prediction.max():.4f}")
        
        # Handle different prediction shapes
        if len(prediction.shape) == 4:
            # Standard case: (batch, height, width, channels)
            binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
        elif len(prediction.shape) == 3:
            # Case: (height, width, channels) or (batch, height, width)
            if prediction.shape[-1] == 1:
                binary_mask = (prediction[0, :, :] > 0.5).astype(np.uint8)
            else:
                binary_mask = (prediction[:, :, 0] > 0.5).astype(np.uint8)
        elif len(prediction.shape) == 2:
            # Case: (height, width)
            binary_mask = (prediction > 0.5).astype(np.uint8)
        else:
            print(f"Unexpected prediction shape: {prediction.shape}")
            return None
        
        print(f"Binary mask shape: {binary_mask.shape}")
        print(f"Binary mask min/max: {binary_mask.min()}/{binary_mask.max()}")
        
        return binary_mask
    except Exception as e:
        import traceback
        print(f"Error in parcel prediction: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def find_contours_and_label(binary_mask):
    """Find contours of individual crop fields and label them"""
    try:
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out very small contours (noise)
        min_area = 100  # Minimum area threshold
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Create labeled mask
        labeled_mask = np.zeros_like(binary_mask)
        crop_fields = []
        
        for i, contour in enumerate(valid_contours):
            # Fill the contour
            cv2.fillPoly(labeled_mask, [contour], i + 1)
            
            # Calculate area and centroid
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            crop_fields.append({
                'id': i + 1,
                'contour': contour,
                'area': area,
                'centroid': (cx, cy)
            })
        
        return labeled_mask, crop_fields
    except Exception as e:
        print(f"Error finding contours: {e}")
        return None, []

def calculate_ndvi_for_fields(original_image, labeled_mask, crop_fields):
    """Calculate NDVI for each crop field"""
    try:
        # Convert RGB to numpy array
        if isinstance(original_image, str):
            img = Image.open(original_image).convert('RGB')
        else:
            img = original_image
        
        img_array = np.array(img)
        
        # For demonstration, we'll simulate NDVI calculation
        # In reality, you would need multispectral data (NIR and Red bands)
        # For now, we'll use a vegetation index approximation from RGB
        
        # Convert to float
        img_float = img_array.astype(np.float32)
        
        # Calculate a simple vegetation index (GNDVI approximation)
        # This is not true NDVI but serves as a proxy
        red = img_float[:, :, 0]
        green = img_float[:, :, 1]
        blue = img_float[:, :, 2]
        
        # Simple vegetation index (Green - Red) / (Green + Red)
        vegetation_index = (green - red) / (green + red + 1e-6)
        
        # Calculate average vegetation index for each crop field
        field_ndvi = []
        for field in crop_fields:
            field_id = field['id']
            field_mask = (labeled_mask == field_id)
            
            if np.any(field_mask):
                avg_ndvi = np.mean(vegetation_index[field_mask])
                field_ndvi.append({
                    'field_id': field_id,
                    'ndvi': avg_ndvi,
                    'area': field['area']
                })
        
        return field_ndvi
    except Exception as e:
        print(f"Error calculating NDVI: {e}")
        return []

def calculate_damage_metrics(field_ndvi_data):
    """Calculate relative damage metrics for each crop field"""
    try:
        if not field_ndvi_data:
            return []
        
        # Extract NDVI values
        ndvi_values = [field['ndvi'] for field in field_ndvi_data]
        
        # Calculate statistics
        mean_ndvi = np.mean(ndvi_values)
        std_ndvi = np.std(ndvi_values)
        
        # Calculate relative damage metrics
        damage_metrics = []
        for field in field_ndvi_data:
            ndvi = field['ndvi']
            
            # Calculate relative damage (lower NDVI = higher damage)
            relative_damage = max(0, (mean_ndvi - ndvi) / (mean_ndvi + 1e-6))
            
            # Classify health status
            if ndvi > mean_ndvi + std_ndvi:
                health_status = "Good"
            elif ndvi > mean_ndvi - std_ndvi:
                health_status = "Moderate"
            else:
                health_status = "Bad"
            
            damage_metrics.append({
                'field_id': field['field_id'],
                'ndvi': ndvi,
                'relative_damage': relative_damage,
                'health_status': health_status,
                'area': field['area']
            })
        
        return damage_metrics
    except Exception as e:
        print(f"Error calculating damage metrics: {e}")
        return []

def create_visualization(original_image, binary_mask, labeled_mask, damage_metrics, crop_fields):
    """Create visualization of results"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original image
        if isinstance(original_image, str):
            img = Image.open(original_image)
        else:
            img = original_image
        
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Satellite Image')
        axes[0, 0].axis('off')
        
        # Predicted binary mask
        axes[0, 1].imshow(binary_mask, cmap='gray')
        axes[0, 1].set_title('Predicted Parcel Boundaries')
        axes[0, 1].axis('off')
        
        # Parcel delineation
        axes[1, 0].imshow(labeled_mask, cmap='tab20')
        axes[1, 0].set_title('Identified Crop Fields')
        axes[1, 0].axis('off')
        
        # Add field numbers and health status
        for field in crop_fields:
            cx, cy = field['centroid']
            field_id = field['id']
            
            # Find corresponding damage metric
            damage_info = next((d for d in damage_metrics if d['field_id'] == field_id), None)
            if damage_info:
                health_color = {'Good': 'green', 'Moderate': 'orange', 'Bad': 'red'}
                color = health_color.get(damage_info['health_status'], 'black')
                axes[1, 0].text(cx, cy, str(field_id), ha='center', va='center', 
                           color=color, fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        # Overlay on original image
        axes[1, 1].imshow(img)
        axes[1, 1].imshow(labeled_mask, alpha=0.3, cmap='tab20')
        axes[1, 1].set_title('Crop Fields Overlay')
        axes[1, 1].axis('off')
        
        # Add field numbers on overlay
        for field in crop_fields:
            cx, cy = field['centroid']
            field_id = field['id']
            
            # Find corresponding damage metric
            damage_info = next((d for d in damage_metrics if d['field_id'] == field_id), None)
            if damage_info:
                health_color = {'Good': 'green', 'Moderate': 'orange', 'Bad': 'red'}
                color = health_color.get(damage_info['health_status'], 'black')
                axes[1, 1].text(cx, cy, str(field_id), ha='center', va='center', 
                           color=color, fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to base64 for web display
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            result = process_satellite_image(filepath)
            
            if result['success']:
                return jsonify(result)
            else:
                return jsonify({'error': result['error']}), 500
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_satellite_image(image_path):
    """Main processing pipeline for satellite image analysis"""
    try:
        # Step 1: Preprocess image
        print("Preprocessing image...")
        image_array = preprocess_image(image_path)
        if image_array is None:
            return {'success': False, 'error': 'Failed to preprocess image'}
        
        # Step 2: Predict parcel delineation
        print("Predicting parcel boundaries...")
        binary_mask = predict_parcel_delineation(image_array)
        if binary_mask is None:
            return {'success': False, 'error': 'Failed to predict parcel boundaries'}
        
        # Step 3: Find contours and label crop fields
        print("Identifying individual crop fields...")
        labeled_mask, crop_fields = find_contours_and_label(binary_mask)
        if not crop_fields:
            return {'success': False, 'error': 'No crop fields detected'}
        
        # Step 4: Calculate NDVI for each field
        print("Calculating NDVI values...")
        field_ndvi_data = calculate_ndvi_for_fields(image_path, labeled_mask, crop_fields)
        
        # Step 5: Calculate damage metrics
        print("Calculating damage metrics...")
        damage_metrics = calculate_damage_metrics(field_ndvi_data)
        
        # Step 6: Create visualization
        print("Creating visualization...")
        visualization = create_visualization(image_path, binary_mask, labeled_mask, damage_metrics, crop_fields)
        
        # Prepare results
        results = {
            'success': True,
            'total_fields': len(crop_fields),
            'crop_fields': damage_metrics,
            'visualization': visualization
        }
        
        return results
        
    except Exception as e:
        import traceback
        print(f"Error in processing pipeline: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return {'success': False, 'error': str(e)}

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    # Load models on startup
    print("Loading models...")
    load_parcel_model()
    print("Starting web application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
