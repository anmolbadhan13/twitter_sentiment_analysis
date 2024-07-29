import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob

# Load the trained sentiment analysis model
model = joblib.load('twitter_sentiment.sav')

# Streamlit app title and CSS styling
st.title('Twitter Sentiment Analysis')
st.image("https://clipartcraft.com/images/white-twitter-logo-icon-8.png",width=100)
st.markdown("""
<style>
    body {
    background-image: url('https://th.bing.com/th/id/OIP.FZpmeDirqwnNzkBv6U8W5wHaEK?rs=1&pid=ImgDetMain');
    background-size: cover;
    color: #333;
    font-family: Arial, sans-serif;
}

.stApp {
    background-color: rgba(255, 255, 255, 0.6);
    padding: 20px;
    border-radius: 10px;
}

h1 {
    color: #ff6347;
    text-align: center;
}

h2 {
    color: #4CAF50;
    text-align: center;
}

.sidebar .sidebar-content {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 10px;
}

.stButton > button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px;
    cursor: pointer;
}

.stButton > button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Input for text to analyze
text_input = st.text_area('Enter text to analyze sentiment')

if st.button('Analyze Sentiment'):
    if text_input:
        # Perform sentiment analysis on the input text
        analysis = TextBlob(text_input)
        sentiment = 'Positive' if analysis.sentiment.polarity > 0 else 'Negative' if analysis.sentiment.polarity < 0 else 'Neutral'

        # Display the result
        st.subheader('Predicted Sentiment')
        if sentiment == 'Positive':
            st.markdown(f"<p class='sentiment-positive'>{text_input} - {sentiment}</p>", unsafe_allow_html=True)
        elif sentiment == 'Neutral':
            st.markdown(f"<p class='sentiment-neutral'>{text_input} - {sentiment}</p>", unsafe_allow_html=True)
        elif sentiment == 'Negative':
            st.markdown(f"<p class='sentiment-negative'>{text_input} - {sentiment}</p>", unsafe_allow_html=True)
    else:
        st.warning('Please enter some text to analyze.')
