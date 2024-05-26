import json
import torch
import torch.nn as nn
from nltk_utils import bag_of_words, tokenize
from nltk.stem import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

def predict_class(sentence, model):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    with torch.no_grad():
        output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    for intent in intents['intents']:
        if tag == intent['tag']:
            return np.random.choice(intent['responses'])

    return "I do not understand..."
