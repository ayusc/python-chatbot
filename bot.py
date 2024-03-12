'''
Copyright (c) 2024, Ayus Chatterjee 
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
'''

import nltk
import random
import os
import pickle
from nltk.chat.util import Chat, reflections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Check if punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Define paths for data storage
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "user_logs.txt")
MODEL_FILE = os.path.join(DATA_DIR, "chatbot_model.pkl")

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load or initialize user logs
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "rb") as f:
        user_logs = pickle.load(f)
else:
    user_logs = []

# Load or initialize chatbot model
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        chatbot_model = pickle.load(f)
else:
    chatbot_model = None

# Define patterns and reflections for the chatbot
patterns = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you?', ['I am doing well, thank you!', 'I\'m fine, thanks for asking.']),
    (r'what is your name?', ['I am a chatbot.', 'You can call me Chatbot.']),
    (r'(.*) your name(.*)', ['My name is Chatbot.', 'I go by the name Chatbot.']),
    (r'(.*) help (.*)', ['I can help you with various topics. Just ask!']),
    (r'(.*) (sorry|apologies)(.*)', ['That\'s alright.', 'No problem.']),
    (r'quit', ['Bye! Take care.', 'Goodbye!']),
]

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Function to update user logs
def update_logs(user_input):
    user_logs.append(user_input)
    with open(LOG_FILE, "wb") as f:
        pickle.dump(user_logs, f)

# Function to train/update chatbot model
def train_chatbot():
    global chatbot_model
    # Preprocess user logs
    preprocessed_logs = [preprocess_text(log) for log in user_logs]
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Vectorize user logs
    X = vectorizer.fit_transform(preprocessed_logs)
    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X, range(len(user_logs)))
    # Generate response based on input
    user_input_vector = vectorizer.transform([preprocess_text(user_logs[-1])])
    predicted_index = classifier.predict(user_input_vector)[0]
    chatbot_model = user_logs[predicted_index]
    # Save updated model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(chatbot_model, f)

# Create a chatbot using the patterns
chatbot = Chat(patterns, reflections)

def main():
    print("Welcome to the chatbot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        update_logs(user_input)
        response = ""
        if chatbot_model:
            response = chatbot_model
        else:
            response = chatbot.respond(user_input)
        print("Chatbot:", response)
        if user_input.lower() == 'quit':
            break
        train_chatbot()

if __name__ == "__main__":
    main()
