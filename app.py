from flask import Flask, render_template, request, jsonify
import torch
import random
import json
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from model import ChatModel
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

stemmer = PorterStemmer()

# Load the intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Preprocess the data
all_words = []
tags = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)

ignore_words = ['?', '!', '.', ',']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
model = ChatModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

def predict_class(sentence, model, device):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    with torch.no_grad():
        output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    logging.debug(f"Predicted tag: {tag}")

    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            logging.debug(f"Response: {response}")
            return response
    return "I do not understand..."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.json.get("msg")
    logging.debug(f"User input: {user_input}")
    response = predict_class(user_input, model, device)
    logging.debug(f"Bot response: {response}")
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
