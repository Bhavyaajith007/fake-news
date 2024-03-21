fakenewsdetectionsourseimport os
import numpy as np
import pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from azure.cognitiveservices.search.websearch import WebSearchClient
from azure.cognitiveservices.search.websearch.models import SafeSearchOptions
from azure.cognitiveservices.search.websearch.models import SafeSearch
import requests

# Azure Text Analytics API credentials
key = "<YOUR_TEXT_ANALYTICS_API_KEY>"
endpoint = "<YOUR_TEXT_ANALYTICS_ENDPOINT>"

# Initialize Text Analytics client
credential = AzureKeyCredential(key)
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

# Load fake news dataset (replace with your file path)
data = pd.read_csv("fake_news_dataset.csv")

# Preprocessing function
def preprocess_text(text):
    # Perform text cleaning here
    return text

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Extract features using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to get trustworthy news source using Bing Search API
def get_trustworthy_source(query):
    subscription_key = "<YOUR_BING_SEARCH_API_KEY>"
    endpoint = "<YOUR_BING_SEARCH_ENDPOINT>"
    
    client = WebSearchClient(endpoint, AzureKeyCredential(subscription_key))
    result = client.web.search(query=query, safe_search=SafeSearchOptions.SAFE)
    
    if result.web_pages.value:
        return result.web_pages.value[0].url
    else:
        return "No trustworthy source found."

# Example usage of get_trustworthy_source function
query = "CNN"
trustworthy_source = get_trustworthy_source(query)
print("Trustworthy Source for", query, ":", trustworthy_source)

# Function to detect fake news
def detect_fake_news(text):
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Extract features
    features = tfidf_vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = clf.predict(features)[0]
    
    return prediction

# Example usage of detect_fake_news function
news_text = "Scientists discover a new cure for COVID-19."
prediction = detect_fake_news(news_text)
print("Prediction for the news:", prediction)
