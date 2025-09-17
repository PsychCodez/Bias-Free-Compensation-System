# Satellite Image Crop Analysis Web Application

A comprehensive web application for analyzing satellite images to identify individual crop fields, calculate NDVI values, and assess crop health with damage metrics.

## Features

- **Parcel Delineation**: Identifies individual crop fields from satellite images using deep learning models
- **NDVI Calculation**: Computes vegetation indices for each crop field
- **Damage Assessment**: Provides relative damage metrics and health classification
- **Interactive Web Interface**: Modern, responsive UI for easy image upload and result visualization
- **Real-time Processing**: Fast analysis pipeline with progress indicators

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd satellite-crop-analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are in place**:
   - Place your trained model files in `Boundary/ParcelDelineation/`
   - The application expects `best-unet-sentinel.hdf5` for parcel delineation

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Upload a satellite image**:
   - Drag and drop an image file onto the upload area
   - Or click "Choose Image" to browse for files
   - Supported formats: JPEG, PNG, TIFF
   - Maximum file size: 16MB

4. **View results**:
   - The application will process your image and display:
     - Total number of identified crop fields
     - Health classification for each field (Good/Moderate/Bad)
     - NDVI values and damage metrics
     - Visual overlay showing field boundaries and numbers

## Project Structure

```
├── app.py                          # Main Flask application
├── templates/
│   └── index.html                  # Web interface template
├── Boundary/
│   └── ParcelDelineation/         # Parcel delineation models and utilities
│       ├── models/                # Model definitions
│       ├── utils/                 # Utility functions
│       └── best-unet-sentinel.hdf5 # Trained model weights
├── uploads/                       # Temporary storage for uploaded images
├── results/                       # Generated analysis results
└── requirements.txt               # Python dependencies
```

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and process satellite image
- `GET /results/<filename>` - Access generated result files

## Processing Pipeline

1. **Image Preprocessing**: Resize and normalize input image
2. **Parcel Delineation**: Use trained U-Net model to identify crop field boundaries
3. **Field Segmentation**: Extract individual crop fields using contour detection
4. **NDVI Calculation**: Compute vegetation indices for each field
5. **Damage Assessment**: Calculate relative damage metrics and health classification
6. **Visualization**: Generate overlay images with field numbers and health indicators

## Model Requirements

The application requires a trained U-Net model for parcel delineation. The model should:
- Accept RGB images of size 224x224
- Output binary segmentation masks
- Be saved in HDF5 format with custom metrics (f1_score, dice_coef_sim)

## Customization

### Adding New Models
To integrate additional models (e.g., for crop classification):

1. Add model loading function in `app.py`
2. Implement prediction function
3. Update the processing pipeline
4. Modify the results display

### Modifying Health Classification
Adjust the health classification thresholds in the `calculate_damage_metrics()` function:

```python
# Current thresholds
if ndvi > mean_ndvi + std_ndvi:
    health_status = "Good"
elif ndvi > mean_ndvi - std_ndvi:
    health_status = "Moderate"
else:
    health_status = "Bad"
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure model file exists and dependencies are installed
2. **Memory Issues**: Reduce image size or use smaller batch sizes
3. **Processing Time**: Large images may take longer to process

### Debug Mode

Run with debug mode enabled for detailed error messages:
```bash
export FLASK_DEBUG=1
python app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using Flask, TensorFlow, and OpenCV
- Parcel delineation models based on U-Net architecture
- UI components from Bootstrap and Font Awesome
