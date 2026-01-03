import pandas as pd
import numpy as np
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

with open('static/model/model.pickle', 'rb') as file:
    model = pickle.load(file)


with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

vocab = pd.read_csv('static/model/vocabulary.txt', header = None)
tokens = vocab[0].tolist()

def preprocessing(text):
    data = pd.DataFrame([text], columns =['tweet'])
    data['tweet'] = data['tweet'].apply(lambda sentence: ' '.join(word.lower() for word in sentence.split()))
    data['tweet'] = data['tweet'].apply(lambda sentence: ' '.join(re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in sentence.split()))
    data['tweet'] = data['tweet'].apply(lambda sentence: re.sub(r'[^\w\s]', '', sentence))
    data['tweet'] = data['tweet'].str.replace('\d+', '', regex = True)
    data['tweet'] = data['tweet'].apply(lambda sentence: ' '.join(word for word in sentence.split() if word not in sw))
    data['tweet'] = data['tweet'].apply(lambda sentence: ' '.join(ps.stem(word) for word in sentence.split()))
    return data['tweet']

def vectorizer(ds):
    vectorized_lst =[]

    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))

        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1

        vectorized_lst.append(sentence_lst)
        
    vectorized_lst_new = np.asarray(vectorized_lst, dtype = np.float32)
    return vectorized_lst_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'Negative'
    else:
        return 'Positive'