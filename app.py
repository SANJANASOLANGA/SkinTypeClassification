from flask import Flask, request, jsonify, render_template, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, static_folder='static')

# Load the trained model
model = load_model('./MobileNetV2.h5')

# Ensure model is ready to make predictions
model.make_predict_function()

recommendations = {
    'Dry': {
        'TeensTwentiesNoAllergies': 'Focus on gentle cleansing and hydration.',
        'ThirtiesFortiesNoAllergies': 'Introduce a hydrating toner and consider adding a moisturizer with SPF and peptides.',
        'FiftiesBeyondNoAllergies': 'Consider adding a retinol serum at night.',
        'AllergiesYes': 'Meet the dermatologist.'
    },
    'Oily': {
        'TeensTwentiesNoAllergies': 'Cleanse twice daily and use oil-free products.',
        'ThirtiesFortiesNoAllergies': 'Continue with oil-free routine and consider adding niacinamide products. Exfoliate 2-3 times a week.',
        'FiftiesBeyondNoAllergies': 'Opt for a gentle cleanser and lightweight moisturizer. Exfoliate once a week.',
        'AllergiesYes': 'Meet the dermatologist.'
    },
    'Normal': {
        'TeensTwentiesNoAllergies': 'Cleanse twice daily, moisturize, and wear sunscreen.',
        'ThirtiesFortiesNoAllergies': 'Introduce a toner and look for a moisturizer with SPF and antioxidants. Exfoliate once a week.',
        'FiftiesBeyondNoAllergies': 'Consider adding a retinol serum at night.',
        'AllergiesYes': 'Meet the dermatologist.'
    },
    'Combination': {
        'TeensTwentiesNoAllergies': 'Use a gentle cleanser, targeted moisturizers, and sunscreen.',
        'ThirtiesFortiesNoAllergies': 'Continue with targeted routine and consider balancing toners/serums. Exfoliate 1-2 times a week.',
        'FiftiesBeyondNoAllergies': 'Adjust moisturizers and consider adding a retinol serum. Exfoliate once a week.',
        'AllergiesYes': 'Meet the dermatologist.'
    },
    'Sensitive': {
        'TeensTwentiesNoAllergies': 'Use fragrance-free, hypoallergenic products and avoid harsh chemicals/scrubs.',
        'ThirtiesFortiesNoAllergies': 'Use fragrance-free, hypoallergenic products and avoid harsh chemicals/scrubs.',
        'FiftiesBeyondNoAllergies': 'Use fragrance-free, hypoallergenic products and avoid harsh chemicals/scrubs.',
        'AllergiesYes': 'Meet the dermatologist.'
    }
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'age' not in request.form or 'allergies' not in request.form:
        return jsonify({'error': 'Missing required input'})

    file = request.files['file']
    age = request.form['age']
    allergies = request.form['allergies']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join('./', file.filename)
        file.save(file_path)

        img = preprocess_image(file_path)

        predictions = model.predict(img)
        class_index = np.argmax(predictions, axis=1)[0]
        classes = ['Dry', 'Oily', 'Normal', 'Combination', 'Sensitive']
        predicted_class = classes[class_index]

        os.remove(file_path)  # Remove the saved file after prediction

        # Determine age group and allergy status
        age_group = ""
        if 13 <= int(age) <= 19 or 20 <= int(age) <= 29:
            age_group = "TeensTwenties"
        elif 30 <= int(age) <= 39 or 40 <= int(age) <= 49:
            age_group = "ThirtiesForties"
        elif 50 <= int(age) <= 59 or int(age) >= 60:
            age_group = "FiftiesBeyond"

        allergy_status = "AllergiesYes" if allergies == "yes" else "NoAllergies"

        # Get recommendations
        recommendation_key = f"{age_group}{allergy_status}"
        recommended_action = recommendations[predicted_class].get(recommendation_key, "Meet the dermatologist.")

        return render_template('index.html', prediction=predicted_class, recommendation=recommended_action, image_file=file.filename)

    return jsonify({'error': 'Something went wrong'})

if __name__ == "__main__":
    app.run(debug=True)
