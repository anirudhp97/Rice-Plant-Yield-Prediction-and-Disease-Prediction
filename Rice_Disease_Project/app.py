from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH='../Rice_Disease_Project/Rice.h5'

model = load_model(MODEL_PATH)
model._make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    preds = model.predict_proba(x)
    return preds
 
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        if preds[0][0] < 0.320236 :
            result="Brownspot Disease spotted!"
        else :
            result="Good News! You're rice plant looks healthy!"
        print(preds)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
        
