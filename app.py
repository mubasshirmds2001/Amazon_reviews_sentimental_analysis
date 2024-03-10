from flask import Flask, render_template, request, jsonify,redirect
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

def analyze_feedbacks(feedbacks):
    cleaned_feedbacks = [preprocess_text(text) for text in feedbacks]
    input_vectors = tfidf_vectorizer.transform(cleaned_feedbacks)
    predicted_sentiments = svm_model.predict(input_vectors)
    return predicted_sentiments

def analyze_text(text):
    cleaned_text = preprocess_text(text)
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    predicted_sentiment = svm_model.predict(input_vector)[0]
    return predicted_sentiment

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        if file:
            feedbacks = []
            for line in file:
                feedbacks.append(line.decode('utf-8').strip())  # Assuming UTF-8 encoding
            predicted_sentiments = analyze_feedbacks(feedbacks)
            sentiment_counts = pd.Series(predicted_sentiments).value_counts()
            colors = ['green', 'red', 'blue']
            plot_url_bar,plot_url_pie = plot_sentiment_distribution(sentiment_counts,colors)
            return jsonify({'plot_url_bar': plot_url_bar, 'plot_url_pie': plot_url_pie})
    else:
        data = request.form['text']
        cleaned_text = preprocess_text(data)
        input_vector = tfidf_vectorizer.transform([cleaned_text])
        predicted_sentiment = svm_model.predict(input_vector)[0]
        plot_url = drawChart(input_vector, predicted_sentiment)
    
        return jsonify({'sentiment': predicted_sentiment, 'plot_url': plot_url})

@app.route('/inputText.html')
def enter_feedback_manually():
    return render_template('inputText.html')

def drawChart(inputVector, predictedSent):
    # Define colors for different sentiments
    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

    # Plot the sentiment along with the input text
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Positive', 'Negative', 'Neutral'], 
                y=[predictedSent.count('Positive'),
                   predictedSent.count('Negative'),
                   predictedSent.count('Neutral')],
                palette=[colors['Positive'], colors['Negative'], colors['Neutral']])
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

@app.route('/upload_file.html')
def upload_file():
    return render_template('upload_file.html')

def plot_sentiment_distribution(sentiment_counts, colors):
    # Pie plot
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution (Pie chart)')
    plt.ylabel('')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_pie = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Bar plot
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=colors)
    plt.title('Sentiment Distribution (Bar Plot)')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_bar = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url_pie, plot_url_bar

if __name__ == '__main__':
    app.run(debug=True)
