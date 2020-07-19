from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import re
import csv
# SCIKIT
from sklearn.preprocessing import OneHotEncoder
# NLTK
from nltk.stem.snowball import SnowballStemmer
# Keras & Tf
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model,model_from_json

with open('./data/tokenizer.pickle', 'rb') as handle:
    tk = pickle.load(handle)
with open('./data/ohe.pickle', 'rb') as hehe:
    onehotenc = pickle.load(hehe)

model=load_model("./data/lstm_3_amzn.h5")
model._make_predict_function()


stopword = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", \
     "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', \
          'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', \
              'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', \
                   "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', \
                       'have', 'has', 'had', 'having', 'do', 'does', 'did', \
                           'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stemmer = SnowballStemmer(language='english')

app = Flask(__name__)

def classify(text):

    text = re.sub('@\S+|https?:\S+|http?:\S|\W+', ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stopword:
            token = stemmer.stem(token)
            tokens.append(token)
    text = " ".join(tokens)
    text = tk.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=100)
    pr = model.predict([text])
    f_score = np.max(pr)
    f_label = onehotenc.inverse_transform(pr)[0][0]
    return f_label.upper(),f_score

class ReviewForm(Form):
    review = TextAreaField("",[validators.DataRequired(),validators.length(min=15)])

@app.route("/")
def index():
    form = ReviewForm(request.form)
    return render_template("reviewform.html", form=form)

@app.route("/results", methods=["POST"])
def results():
    form = ReviewForm(request.form)
    if request.method == "POST":
        review = request.form["review"]
    else:
        review = "[no review]"
    y, proba = classify(review)
    return render_template("results.html",content=review,prediction=y,probability=round(proba*100, 2))
    return render_template("reviewform.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)
