from flask import Flask, render_template, request, redirect
from prediction_pipeline import preprocessing, vectorizer, get_prediction
from logger import logging

app = Flask(__name__)

logging.info("Application started")

data = dict()
reviews = []
positive = 0
negative = 0

@app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative

    logging.info("Open Home page")

    return render_template('index.html', data=data)

@app.route("/", methods=['post'])
def my_post():
    text = request.form['text']
    logging.info(f"text: {text}")

    preprocessd_text = preprocessing(text)
    logging.info(f"preprocessed text: {preprocessd_text}")

    vectorized_text = vectorizer.transform([preprocessd_text])
    logging.info(f"vectorized text: {vectorized_text}")
    
    prediction = get_prediction(vectorized_text)
    logging.info(f"prediction: {prediction}")

    if prediction == 'Negative':
        global negative
        negative += 1
    else:
        global positive
        positive += 1

    reviews.insert(0, text)
    return redirect(request.url)


if __name__ == "__main__":
    app.run()