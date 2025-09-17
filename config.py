"""
Configuration settings for the Satellite Image Crop Analysis Web Application
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Flask configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    RESULTS_FOLDER = BASE_DIR / 'results'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Model paths
    PARCEL_MODEL_PATH = BASE_DIR / 'Boundary' / 'ParcelDelineation' / 'best-unet-sentinel.hdf5'
    
    # Image processing settings
    IMAGE_SIZE = (224, 224)
    MIN_FIELD_AREA = 100  # Minimum area for valid crop fields
    
    # NDVI calculation settings
    NDVI_THRESHOLDS = {
        'good': 0.6,
        'moderate': 0.3,
        'bad': 0.0
    }
    
    # Health classification thresholds (relative to mean)
    HEALTH_THRESHOLDS = {
        'good_std_multiplier': 1.0,    # Above mean + 1*std
        'moderate_std_multiplier': -1.0  # Above mean - 1*std
    }

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
