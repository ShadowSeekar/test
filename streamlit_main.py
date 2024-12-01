import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load the model using TFSMLayer
model_path = "tf_model"  # Path to the SavedModel folder
model = TFSMLayer(model_path, call_endpoint="serving_default")

# Load the tokenizer (ensure tokenizer.pkl is saved alongside the model)
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
        # Get the prediction (use model like a regular layer)
        prediction = model(processed_text).numpy()[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        st.write(f"**Sentiment Score:** {prediction:.2f}")
        st.write(f"**Sentiment:** {sentiment}")
    else:
        st.write("Please enter valid text.")
