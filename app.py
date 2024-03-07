from flask import Flask, render_template, request, jsonify
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64
from sklearn.feature_extraction.text import TfidfVectorizer

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
    return render_template('inputText.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.form['text']
    cleaned_text = preprocess_text(data)
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    predicted_sentiment = svm_model.predict(input_vector)[0]
    plot_url = classify_and_visualize_text(input_vector, predicted_sentiment)
    
    return jsonify({'sentiment': predicted_sentiment, 'plot_url': plot_url})

def classify_and_visualize_text(inputVector, predictedSentiment):
    # Plot the sentiment along with the input text
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Positive', 'Negative', 'Neutral'], y=[predictedSentiment.count('Positive'),
                                                          predictedSentiment.count('Negative'),
                                                          predictedSentiment.count('Neutral')])
    plt.title('Predicted Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

if __name__ == '__main__':
    app.run(debug=True)


