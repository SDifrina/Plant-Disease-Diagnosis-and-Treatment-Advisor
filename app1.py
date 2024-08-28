import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
model_path = r"/home/vaish/tens/my_modell.h5"
model = tf.keras.models.load_model(model_path, compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')
# Read the CSV file without parsing 'Disease' as dates
data = pd.read_csv(r"/home/vaish/tens/Dataset.csv", index_col='Disease')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds

def get_treatment_for_disease(disease_name):
    try:
        treatment = data.loc[disease_name, 'Treatment']
        return treatment
    except KeyError:
        return "Disease not found in the dataset"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return jsonify({'error': 'Unsupported file format'})

        # Ensure that the uploads directory exists
        upload_dir = r"/home/vaish/tens/uploads/"
        os.makedirs(upload_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        preds = model_predict(file_path, model)
        # Replace 'disease_class' with your list of disease classes
        disease_class = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
                         

        predicted_class_index = np.argmax(preds)
        if predicted_class_index < len(disease_class):
            predicted_class = disease_class[predicted_class_index]
            print(f"Predicted Class Index: {predicted_class_index}")
            print(f"Predicted Class: {predicted_class}")
            treatment = get_treatment_for_disease(predicted_class)
            return render_template('result.html', result1=predicted_class, result2=treatment)
        else:
            print("Disease not found")
            return jsonify({'error': 'Disease not found'})

    return render_template('ind.html')

if __name__ == '__main__':
    app.run(debug=True)

