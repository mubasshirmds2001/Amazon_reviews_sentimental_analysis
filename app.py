from flask import Flask, render_template, request, jsonify, send_file
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
svm_model = joblib.load("Models/sentiment_model.pkl")
tfidf_vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")

# Load the dataset for sentiment distribution
df = pd.read_csv("amazon_reviews.csv")

# Preprocess text data
def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert text to lowercase
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.form['text']
    cleaned_text = preprocess_text(data)
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    predicted_sentiment = svm_model.predict(input_vector)[0]
    
    return jsonify({'sentiment': predicted_sentiment})

@app.route('/sentiment-graph', methods=['GET', 'POST'])
def sentiment_graph():
    if request.method == 'POST':
        data = request.json['text']
        cleaned_text = preprocess_text(data)
        polarity_score = TextBlob(cleaned_text).sentiment.polarity
        sentiment_data = {'Positive': max(0, polarity_score), 'Negative': max(0, -polarity_score), 'Neutral': 0.0}

        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(sentiment_data.keys()), y=list(sentiment_data.values()))
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        
        # Save the plot as an image file
        plot_filename = 'sentiment_plot.png'
        plt.savefig(plot_filename)

        # Return the image file to the client
        return send_file(plot_filename, mimetype='image/png')

    # Handle GET requests (if needed)
    return render_template('sentiment_graph.html')


if __name__ == '__main__':
    app.run(debug=True)
