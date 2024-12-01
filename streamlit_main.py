import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained LSTM model (SavedModel format)
model_directory = "tf_model"  # Replace with your actual directory
model = tf.keras.models.load_model(model_directory)

# Load the tokenizer (if it was saved separately)
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Function to preprocess text
def preprocess_text(text, tokenizer, max_len=100):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len)
    return padded

# Streamlit app
st.title("Sentiment Analysis with LSTM")

st.write("Enter text to analyze its sentiment:")

user_input = st.text_area("Text Input", placeholder="Type your text here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Preprocess input text
        processed_text = preprocess_text(user_input, tokenizer)
        # Get the prediction
        prediction = model.predict(processed_text)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        st.write(f"**Sentiment Score:** {prediction:.2f}")
        st.write(f"**Sentiment:** {sentiment}")
    else:
        st.write("Please enter valid text.")
