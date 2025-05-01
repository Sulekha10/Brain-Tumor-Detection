from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model('BrainTumor_VGG16.h5')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    img_path = None

    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        img_path = filepath

        img = image.load_img(filepath, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        result = model.predict(img)
        predicted_class = np.argmax(result)

        if predicted_class == 0:
            prediction = "No Brain Tumor Detected"
        else:
            prediction = "Brain Tumor Detected"

    return render_template('index.html', prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
