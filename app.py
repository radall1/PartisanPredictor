import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load label encoder

app = Flask(__name__)

def predict(text):
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('model/tokenizer.json', 'r') as f:
        tokenizer_config = f.read()
    tokenizer = tokenizer_from_json(tokenizer_config)

    max_sequence_length = 100 
    model_path = 'model/political_model_v2.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    predicted_probabilities = model.predict(padded_sequence)
    predicted_label_index = np.argmax(predicted_probabilities)
    predicted_label = label_encoder.classes_[predicted_label_index] if label_encoder else predicted_label_index
    return predicted_label, str(int(predicted_probabilities.max()*100))

@app.route('/', methods=['POST'])
def handle_input():
    user_input = request.json.get('input', '')
    label, percent = predict(user_input)
    label = 'Democrat' if label == 'D' else 'Republican'
    output = "You are a " + label + " - " + percent + "% confident.\n"
    return jsonify({'output': output})
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
