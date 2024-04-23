import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load label encoder

app = Flask(__name__)

def predict(texts):
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('tokenizer.json', 'r') as f:
        tokenizer_config = f.read()
    tokenizer = tokenizer_from_json(tokenizer_config)

    max_sequence_length = 100 
    model_path = 'political_model_v2.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    predicted_probabilities = model.predict(padded_sequences)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predicted_probabilities, axis=1))
    return predicted_labels, np.max(predicted_probabilities, axis=1)

@app.route('/', methods=['POST'])
def handle_input():
    user_input = request.json.get('input', '')
    output = predict(user_input)  # Implement your magic function here
    return jsonify({'output': output})

def doMagic(user_input):
    # Implement your magic function here
    # For demonstration purposes, echo back the input
    return "You said: " + user_input
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
