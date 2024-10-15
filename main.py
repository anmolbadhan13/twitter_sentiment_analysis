import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd  # For saving feedback to a CSV file
from PIL import Image  # To load the Twitter icon

# Download stopwords for preprocessing
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Load the Logistic Regression model
model = pickle.load(open('twitter_sentiment_log.sav', 'rb'))

# Load the TfidfVectorizer
vectorizer = pickle.load(open('twitter_vectorizer.sav', 'rb'))


# Define the text preprocessing function
def stem(content):
    content = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabet characters
    content = content.lower()  # Convert to lowercase
    content = content.split()  # Split into words
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]  # Remove stopwords & stem
    return ' '.join(content)  # Join back into a string


# Function to predict sentiment
def predict_sentiment(text):
    processed_text = stem(text)
    text_vector = vectorizer.transform([processed_text])

    # Get probabilities for each class
    probabilities = model.predict_proba(text_vector)[0]  # Get probabilities for both classes
    positive_prob = probabilities[1]  # Probability of positive class
    negative_prob = probabilities[0]  # Probability of negative class
    neutral_prob = 1 - (positive_prob + negative_prob)  # Neutral probability

    # Define thresholds for classifying sentiment
    if positive_prob > 0.6:  # Positive
        sentiment = "Positive"
    elif negative_prob > 0.6:  # Negative
        sentiment = "Negative"
    else:  # Neutral
        sentiment = "Neutral"

    return sentiment  # Return only the sentiment


# Define custom CSS for the main page, sidebar, and text input
page_bg_style = '''
<style>
.stApp {
    background-color: #e6f7ff;
    color: #333;
    font-family: "Helvetica Neue", sans-serif;
}
h1 {
    color: #003366;
    text-align: center;
    font-weight: bold;
    margin-bottom: 20px;
}
textarea {
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
    background-color: #f0f8ff;
    color: #003366;
    border: 1px solid #005c99;
}
button {
    background-color: #005c99;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}
button:hover {
    background-color: #004080;
}

/* Styling for the sidebar */
.sidebar .sidebar-content {
    background-color: #d9eaf7;
    color: #333;
}
.sidebar h1, .sidebar h2, .sidebar h3 {
    color: #003366;
}
.sidebar .stTextInput, .sidebar .stTextArea {
    border-radius: 10px;
    padding: 8px;
    font-size: 14px;
    background-color: #f0f8ff;
    color: #003366;
    border: 1px solid #005c99;
}
.sidebar .stButton button {
    background-color: #005c99;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 14px;
    cursor: pointer;
}
.sidebar .stButton button:hover {
    background-color: #004080;
}
</style>
'''

# Apply custom CSS
st.markdown(page_bg_style, unsafe_allow_html=True)


# Streamlit web app
def main():
    # Display the Twitter icon
    twitter_icon = Image.open('twitter_icon.png')  # Ensure the file path is correct
    st.image(twitter_icon, width=50)

    st.title("Twitter Sentiment Analysis")

    # Sidebar for Feedback and Instructions
    st.sidebar.header("User Feedback")
    st.sidebar.write("Please provide your feedback below:")
    with st.sidebar.form(key='feedback_form'):
        name = st.text_input("Name", placeholder="Enter your name")
        email = st.text_input("Email", placeholder="Enter your email")
        feedback = st.text_area("Feedback", placeholder="Write your feedback here")
        submit_button = st.form_submit_button("Submit Feedback")

        if submit_button:
            try:
                feedback_data = pd.DataFrame([[name, email, feedback]], columns=['Name', 'Email', 'Feedback'])
                feedback_data.to_csv('feedback.csv', mode='a', header=False, index=False)
                st.sidebar.success("Thank you for your feedback!")
            except Exception as e:
                st.sidebar.error(f"An error occurred while saving feedback: {e}")

    st.write("Enter a tweet or text to analyze its sentiment.")

    # Text area for user input
    user_input = st.text_area("Enter tweet or text:", "", height=150)

    if st.button("Analyze Sentiment"):
        if user_input:
            with st.spinner("Analyzing sentiment..."):
                sentiment = predict_sentiment(user_input)

                # Display sentiment result with color coding
                if sentiment == "Negative":
                    st.markdown(
                        '<p style="color: red; font-size: 20px; font-weight: bold;">Predicted Sentiment: Negative</p>',
                        unsafe_allow_html=True)
                elif sentiment == "Positive":
                    st.markdown(
                        '<p style="color: green; font-size: 20px; font-weight: bold;">Predicted Sentiment: Positive</p>',
                        unsafe_allow_html=True)
                else:  # Neutral
                    st.markdown(
                        '<p style="color: orange; font-size: 20px; font-weight: bold;">Predicted Sentiment: Neutral</p>',
                        unsafe_allow_html=True)

        else:
            st.warning("Please enter some text to analyze.")


# Define custom CSS for the footer
footer_style = '''
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #004080;
    color: white;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
}

.footer a {
    color: #f0f8ff;
    text-decoration: none;
    margin: 0 10px;
}

.footer a:hover {
    text-decoration: underline;
    color: #cce7ff;
}
</style>
'''

# Apply the footer style
st.markdown(footer_style, unsafe_allow_html=True)

# Add content to the footer
footer = '''
<div class="footer">
    <p>Made by Anmol Badhan</p>
    <p>
        <a href="https://github.com/anmolbadhan13" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/anmol-badhan-770434320/" target="_blank">LinkedIn</a> |
        <a href="mailto:badhananmol24@gmail.com">Email</a>
    </p>
    <p>&copy; 2024 tweets analysis (nlp)</p>
</div>
'''

# Display the footer
st.markdown(footer, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()


