from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('./model_VGG16_23mei.h5')

# Ensure model is ready to make predictions
model.make_predict_function()


if __name__ == "__main__":
    app.run(debug=True)
