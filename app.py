from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import numpy as np
import os

# app = Flask(__name__)
app = Flask(__name__, static_folder='static')
model = YOLO(r'runs\classify\train2\weights\best2.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/aboutUs')
def aboutUs():
    return render_template('aboutUs.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST']) 
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    image_file = request.files['image']
    image_path = os.path.join('./uploads', image_file.filename)
    image_file.save(image_path)

    results = model(image_path)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    max_prob = max(probs)
    if max_prob > 0.80:
        predicted_class = names_dict[np.argmax(probs)]
        return jsonify({'predicted_class': predicted_class})
    else:
        return jsonify({'predicted_class': 'undefined'})

if __name__ == '__main__':
    app.run(debug=True)
