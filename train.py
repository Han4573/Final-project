import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from model import ChatModel

nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Tokenize and stem the words in the patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = np.zeros(len(all_words), dtype=np.float32)
    sentence_words = [stemmer.stem(w.lower()) for w in pattern_sentence]
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

model = ChatModel(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model(torch.from_numpy(X_train).float())
    loss = criterion(outputs, torch.from_numpy(y_train).long())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete.')
torch.save(model.state_dict(), "model.pth")
