from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('rice_leaf_disease_model.h5')

data_dir = 'rice_leaf_diseases'
categories = os.listdir(data_dir)


def prepare_image(image, target_size=(64, 64)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = Image.open(file.stream)
            prepared_image = prepare_image(image)
            prediction = model.predict(prepared_image)
            predicted_class = categories[np.argmax(prediction)]
            return render_template('result.html', prediction=predicted_class)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
