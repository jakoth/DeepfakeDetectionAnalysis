from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
CORS(app)  # Add CORS(app) to enable CORS support

# Load the trained model
model = load_model('deepfake_detection_model.h5')

# Preprocessing function for images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

# Endpoint for receiving image and predicting
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the image to a temporary file (if needed)
    # image_path = 'temp_image.jpg'
    # file.save(image_path)

    # Or directly process the image from memory
    img_array = preprocess_image(file)

    # Make prediction
    prediction = model.predict(img_array)
    if prediction < 0.5:
        result = "REAL"
    else:
        result = "FAKE (Deepfake)"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
