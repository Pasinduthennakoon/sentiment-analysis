import pandas as pd
import numpy as np
import re
import string
import pickle

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

with open('static/model/model.pickle', 'rb') as file:
    model = pickle.load(file)


with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

with open('static/model/vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)
def preprocessing(text):
    # Convert to lowercase and split
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove punctuation and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in sw]
    return " ".join(words)

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'Negative'
    else:
        return 'Positive'