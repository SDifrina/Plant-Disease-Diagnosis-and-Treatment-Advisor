import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Change Matplotlib backend to 'Agg'
import matplotlib
matplotlib.use('Agg')

# Assuming 'model' and 'img' are defined
loaded_model = tf.keras.models.load_model('my_modell.h5')
print("model saved and loaded!!!!")

# Load and preprocess the image
img_path = r"/home/vaish/tens/uploads/t.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Predict the class probabilities
probs = loaded_model.predict(img)[0]

class_names = ['Apple___Apple_scab',
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
 

# Get the predicted class index
pred_class_prob = np.argmax(probs)

# Check if the predicted class index is within a valid range
if 0 <= pred_class_prob < len(class_names):
    pred_class_name = class_names[pred_class_prob]
else:
    pred_class_name = "Unknown Class"

# Print the predicted class name and probability
print(f'Predicted class: {pred_class_name}')
print(f'Probability: {probs[pred_class_prob] if 0 <= pred_class_prob < len(probs) else 0.0}')

# Display the image with the predicted class and probability
plt.figure(figsize=(10, 10))
plt.imshow(img[0] / 255.)
plt.axis('off')
plt.text(10, 20, f'Predicted class: {pred_class_name}\nProbability: {probs[pred_class_prob] if 0 <= pred_class_prob < len(probs) else 0.0:.2f}', fontsize=20, color='red', bbox=dict(facecolor='white', alpha=0.8))

# Save the plot as an image file (optional)
plt.savefig('predicted_image.png')

# Finally, no need to call plt.show() in this context

