from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load pre-trained model
nlp = pipeline("conversational")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = nlp(user_input)
    return jsonify({'response': response['generated_text']})

if __name__ == '__main__':
    app.run(debug=True)
