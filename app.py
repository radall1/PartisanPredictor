from flask import Flask, request, jsonify, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def handle_input():
    user_input = request.json.get('input', '')
    output = doMagic(user_input)  # Implement your magic function here
    return jsonify({'output': output})

def doMagic(user_input):
    # Implement your magic function here
    # For demonstration purposes, echo back the input
    return "You said: " + user_input

if __name__ == '__main__':
    app.run(debug=True)
