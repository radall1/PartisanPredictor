import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_config = f.read()
tokenizer = tokenizer_from_json(tokenizer_config)

# Load model
max_sequence_length = 100  # Adjust according to your model's input shape
model_path = 'political_model_v2.h5'
model = tf.keras.models.load_model(model_path, compile=False)

def predict(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    predicted_probabilities = model.predict(padded_sequences)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predicted_probabilities, axis=1))
    return predicted_labels, np.max(predicted_probabilities, axis=1)

def predict2(text):
    return 'lol'
    
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_input():
    data = request.json.get('data', [])
    if not isinstance(data, list):
        return jsonify({'error': 'Invalid input format'}), 400

    predictions = predict2(data)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=False)
