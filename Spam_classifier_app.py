import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the trained model and tokenizer
model = tf.keras.models.load_model('Spam_classifier_model.keras', compile=False)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define max length (same as training)
max_len = 100
stop_words = set(stopwords.words('english'))

# Helper function to clean text
def clean_text(text):
    # Lowercase, remove punctuation, remove stopwords
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Streamlit App
st.title("📧 Email Spam Classifier")
st.write("Predict whether an email is spam or not using a trained LSTM model.")

# Email input
user_input = st.text_area("Enter the email text below:")

if st.button("Predict"):  # Only predict when the button is clicked
    # Preprocess the input text
    cleaned_text = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Make prediction
    # Convert logits to probabilities
    prediction = model.predict(padded)[0][0]
    probability = 1 / (1 + np.exp(-prediction))  # Apply sigmoid
    result = "🟢 Not Spam" if probability < 0.5 else "🔴 Spam"
    confidence = round(probability * 100, 2) if probability >= 0.5 else round((1 - probability) * 100, 2)

    # Display the result
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {confidence}%")
