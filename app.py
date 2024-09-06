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
        'TeensTwentiesNoAllergies': {
            'Routine': 'Gentle cleansing, hydrating moisturizer, sunscreen.',
            'Tips': 'Drink plenty of water, avoid hot showers, consider a humidifier.'
        },
        'ThirtiesFortiesNoAllergies': {
            'Routine': 'Gentle cleansing, hydrating toner, moisturizer with SPF, weekly exfoliation.',
            'Tips': 'Incorporate a hyaluronic acid serum for intense hydration, use a gentle exfoliant like a chemical peel.'
        },
        'FiftiesBeyondNoAllergies': {
            'Routine': 'Gentle cleansing, hydrating serum, moisturizer with SPF, weekly exfoliation, retinol at night.',
            'Tips': 'Consider using a gentle eye cream to address fine lines and wrinkles, avoid harsh products that can irritate sensitive skin.'
        },
        'AllergiesYes': {
            'Routine': 'Use fragrance-free, hypoallergenic products, avoid harsh chemicals and scrubs.',
            'Tips': 'Patch test new products before applying them to your face, consult a dermatologist for allergy testing.'
        }
    },
    'Oily': {
        'TeensTwentiesNoAllergies': {
            'Routine': 'Twice-daily cleansing, oil-free moisturizer, sunscreen, weekly exfoliation.',
            'Tips': 'Avoid heavy makeup and greasy foods, use blotting papers to control shine.'
        },
        'ThirtiesFortiesNoAllergies': {
            'Routine': 'Twice-daily cleansing, oil-free moisturizer, sunscreen, niacinamide serum, 2-3 times weekly exfoliation.',
            'Tips': 'Consider using a clay mask to absorb excess oil, avoid over-washing, which can lead to more oil production.'
        },
        'FiftiesBeyondNoAllergies': {
            'Routine': 'Gentle cleansing, lightweight moisturizer, sunscreen, weekly exfoliation.',
            'Tips': 'As skin naturally becomes less oily with age, adjust your routine accordingly.'
        },
        'AllergiesYes': {
            'Routine': 'Use fragrance-free, hypoallergenic products, avoid harsh chemicals and scrubs.',
            'Tips': 'Patch test new products before applying them to your face, consult a dermatologist for allergy testing.'
        }
    },
    'Normal': {
        'TeensTwentiesNoAllergies': {
            'Routine': 'Twice-daily cleansing, moisturizer, sunscreen.',
            'Tips': 'Maintain a balanced diet and adequate hydration.'
        },
        'ThirtiesFortiesNoAllergies': {
            'Routine': 'Twice-daily cleansing, hydrating toner, moisturizer with SPF, weekly exfoliation.',
            'Tips': 'Consider adding a retinol serum at night for anti-aging benefits.'
        },
        'FiftiesBeyondNoAllergies': {
            'Routine': 'Gentle cleansing, hydrating serum, moisturizer with SPF, weekly exfoliation, retinol at night.',
            'Tips': 'Use a gentle eye cream to address fine lines and wrinkles.'
        },
        'AllergiesYes': {
            'Routine': 'Use fragrance-free, hypoallergenic products, avoid harsh chemicals and scrubs.',
            'Tips': 'Patch test new products before applying them to your face, consult a dermatologist for allergy testing.'
        }
    },
    'Combination': {
        'TeensTwentiesNoAllergies': {
            'Routine': 'Twice-daily cleansing, targeted moisturizer, sunscreen, weekly exfoliation.',
            'Tips': 'Use a clay mask for oily areas and a hydrating mask for dry areas.'
        },
        'ThirtiesFortiesNoAllergies': {
            'Routine': 'Twice-daily cleansing, balancing toner, targeted moisturizer, sunscreen, 1-2 times weekly exfoliation.',
            'Tips': 'Consider using a dual-action cleanser that addresses both oily and dry areas.'
        },
        'FiftiesBeyondNoAllergies': {
            'Routine': 'Gentle cleansing, targeted moisturizer, sunscreen, weekly exfoliation.',
            'Tips': 'As skin naturally becomes less oily with age, adjust your routine accordingly.'
        },
        'AllergiesYes': {
            'Routine': 'Use fragrance-free, hypoallergenic products, avoid harsh chemicals and scrubs.',
            'Tips': 'Patch test new products before applying them to your face, consult a dermatologist for allergy testing.'
        }
    },
    'Sensitive': {
        'TeensTwentiesNoAllergies': {
            'Routine': 'Gentle cleansing, hypoallergenic moisturizer, sunscreen.',
            'Tips': 'Avoid harsh chemicals, fragrances, and alcohol-based products.'
        },
        'ThirtiesFortiesNoAllergies': {
            'Routine': 'Gentle cleansing, hypoallergenic moisturizer, sunscreen.',
            'Tips': 'Consider using a calming serum with ingredients like chamomile or aloe vera.'
        },
        'FiftiesBeyondNoAllergies': {
            'Routine': 'Gentle cleansing, hypoallergenic moisturizer, sunscreen.',
            'Tips': 'Use a gentle eye cream to address fine lines and wrinkles.'
        },
        'AllergiesYes': {
            'Routine': 'Use fragrance-free, hypoallergenic products, avoid harsh chemicals and scrubs.',
            'Tips': 'Patch test new products before applying them to your face, consult a dermatologist for allergy testing.'
        }
    }
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/skin_types')
def skin_types():
    return render_template('skin_types.html')

@app.route('/identify')
def identify():
    return render_template('identify.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

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
    if 13 <= int(age) <= 29:
        age_group = "TeensTwenties"
    elif 30 <= int(age) <= 49:
        age_group = "ThirtiesForties"
    elif int(age) >= 50:
        age_group = "FiftiesBeyond"

    allergy_status = "AllergiesYes" if allergies == "yes" else "NoAllergies"

    if allergy_status == "AllergiesYes":
        recommendation_key = "AllergiesYes"
    else:
        recommendation_key = f"{age_group}NoAllergies"

    recommended_action = recommendations[predicted_class].get(recommendation_key, {})
    routine = recommended_action.get('Routine', '')
    tips = recommended_action.get('Tips', '')

    print(f"Debug - Recommendation Key: {recommendation_key}")
    print(f"Debug - Recommended Action: {recommended_action}")
    print(f"Debug - Routine: {routine}")
    print(f"Debug - Tips: {tips}")

    return render_template('identify.html', prediction=predicted_class, routine=routine, tips=tips, image_file=file.filename)

    return jsonify({'error': 'Something went wrong'})

if __name__ == "__main__":
    app.run(debug=True)