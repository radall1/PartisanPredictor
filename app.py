from tensorflow.keras.preprocessing.text import tokenizer_from_json
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open('/Users/mohamed/Downloads/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('Users/mohamed/Downloads/tokenizer.json', 'r') as f:
        tokenizer_config = f.read()
tokenizer = tokenizer_from_json(tokenizer_config)
    
max_sequence_length = 100 # Adjust according to your model's input shape
model_path = '/Users/mohamed/Downloads/political_model_v2.h5'
model = tf.keras.models.load_model(model_path)

def predict(text):
    if tokenizer is None or model is None:
        raise ValueError("Tokenizer or model is not loaded. Please call load_model_and_tokenizer first.")
    
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    predicted_probabilities = model.predict(padded_sequence)
    predicted_label_index = np.argmax(predicted_probabilities)
    predicted_label = label_encoder.classes_[predicted_label_index] if label_encoder else predicted_label_index
    return predicted_label, predicted_probabilities.max()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def handle_input():
    user_input = request.json.get('input', '')
    output = predict(user_input)  # Implement your magic function here
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)

