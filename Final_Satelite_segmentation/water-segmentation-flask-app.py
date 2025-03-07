from flask import Flask, render_template, request, send_file
import os
import numpy as np
import tifffile
from PIL import Image
import tensorflow as tf
from io import BytesIO
import uuid
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
PREVIEW_FOLDER = 'previews'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['PREVIEW_FOLDER'] = PREVIEW_FOLDER

# Load your pre-trained model
model = None

def load_model():
    global model
    # Replace this with your actual model loading code
    # model = tf.keras.models.load_model('path_to_your_model')
    # For demonstration purposes:
    model = tf.keras.models.load_model('Satelite_detection.h5')

def create_preview_image(tiff_path):
    """Create a PNG preview of the last layer of the TIFF image."""
    # Load the TIFF image using tifffile
    tiff_data = tifffile.imread(tiff_path)
    
    # Extract the last band (band 11, index 10)
    last_band = tiff_data[..., 10] if tiff_data.ndim >= 3 else tiff_data
    
    # Normalize for visualization (adjust based on your data range)
    normalized = last_band.astype(np.float32)
    min_val = np.min(normalized)
    max_val = np.max(normalized)
    if max_val > min_val:
        normalized = (normalized - min_val) / (max_val - min_val) *255
    normalized = normalized.astype(np.uint8)
    
    # Create a preview image
    preview_img = Image.fromarray(normalized)
    
    # Generate a filename for the preview
    preview_filename = os.path.basename(tiff_path).replace('.tiff', '_preview.png').replace('.tif', '_preview.png')
    preview_path = os.path.join(app.config['PREVIEW_FOLDER'], preview_filename)
    
    # Save the preview image
    preview_img.save(preview_path)
    
    return preview_filename

def preprocess_image(tiff_path):
    """Preprocess the TIFF image for the model using tifffile and select bands 7,8,9,10,11."""
    # Load the TIFF image using tifffile
    tiff_data = tifffile.imread(tiff_path)
    
    # Check if the image has enough bands
    if tiff_data.ndim < 3 or tiff_data.shape[-1] < 11:
        raise ValueError(f"The TIFF file doesn't have enough bands. Expected at least 11 bands, got shape {tiff_data.shape}")
    
    # Select bands 7,8,9,10,11 (indices 6,7,8,9,10 since zero-indexed)
    selected_bands = tiff_data[..., [6, 7, 8, 9, 10]]
    
    # Normalize the data (adjust normalization based on your specific data)
    image = selected_bands.astype(np.float32)
    #image = image / np.max(image)  # Simple normalization
    
    # Add batch dimension if needed
    image = np.expand_dims(image, axis=0)
    return image

def segment_water(image):
    """Run water segmentation on the preprocessed image."""
    # Ensure the model is loaded
    if model is None:
        load_model()
    
    # Predict water segmentation
    prediction = model.predict(image)
    
    # Convert prediction to binary mask
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    return binary_mask[0, :, :, 0]  # Remove batch and channel dimensions

def save_result(binary_mask, filename):
    """Save the segmentation result as a PNG."""
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    result_img = Image.fromarray(binary_mask)
    result_img.save(result_path)
    return result_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file and file.filename.lower().endswith('.tiff') or file.filename.lower().endswith('.tif'):
        # Generate unique filename
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        tiff_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(tiff_path)
        
        try:
            # Create a preview image of the last layer
            preview_filename = create_preview_image(tiff_path)
            
            # Process the image for segmentation
            preprocessed_image = preprocess_image(tiff_path)
            water_mask = segment_water(preprocessed_image)
            
            # Save and return the result
            result_filename = f"result_{unique_filename.replace('.tiff', '.png').replace('.tif', '.png')}"
            result_path = save_result(water_mask, result_filename)
            
            # Get current date/time for the template
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return render_template('result.html', 
                                  original_filename=file.filename,
                                  preview=preview_filename, 
                                  result=result_filename,
                                  now=now)
        
        except Exception as e:
            return render_template('index.html', error=f'Error processing image: {str(e)}')
    
    return render_template('index.html', error='File must be a TIFF image')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/previews/<filename>')
def preview_file(filename):
    return send_file(os.path.join(app.config['PREVIEW_FOLDER'], filename))

@app.route('/results/<filename>')
def result_file(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

if __name__ == '__main__':
    load_model()
    app.run(debug=True)