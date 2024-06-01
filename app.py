import os
from flask import Flask, request, jsonify, render_template, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('./model_VGG16_23mei.h5')

# Ensure model is ready to make predictions
model.make_predict_function()

# Recommendations dictionary
recommendations = {
    'Dry': 'Use moisturizer',
    'Oily': 'Use suncream',
    'Normal': 'Follow the normal routine',
    'Combination': '1. Follow normal routine\n2. Meet the doctor',
    'Sensitive': '1. Follow normal routine\n2. Meet the doctor'
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file to a temporary directory
        file_path = os.path.join('./uploads', file.filename)
        file.save(file_path)
        
        img = preprocess_image(file_path)
        predictions = model.predict(img)
        class_index = np.argmax(predictions, axis=1)[0]
        classes = ['Dry', 'Oily', 'Normal', 'Combination', 'Sensitive']
        predicted_class = classes[class_index]


        print("Predicted class:", predicted_class)  # Debugging line

        os.remove(file_path)  # Remove the saved file after prediction

        
        print("Predicted class:", predicted_class)  # Debugging line

        os.remove(file_path)  # Remove the saved file after prediction

        # Get recommendations
        recommended_action = recommendations[predicted_class]
        
        return render_template('index.html', prediction=predicted_class, recommendation=recommended_action, image_file=file.filename)
    
    return jsonify({'error': 'Something went wrong'})

if __name__ == "__main__":
    # Create the uploads directory if it doesn't exist
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')
    app.run(debug=True)
