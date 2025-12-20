import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import sys
from tensorflow.keras import preprocessing
sys.modules['keras.src.preprocessing'] = preprocessing
import re
import nltk
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords')

model_path = "tf_model"  
model = TFSMLayer(model_path, call_endpoint="serving_default")

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    '''Cleans text data up, leaving only 2 or more char long non-stopwords composed of A-Z & a-z only in lowercase'''
    sentence = sen.lower()
    sentence = remove_tags(sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence) 
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence) 
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

# Streamlit app
st.title("Sentiment Analysis with LSTM")

st.write("Enter text to analyze its sentiment:")

user_input = st.text_area("Text Input", placeholder="Type your text here...")

if st.button("Analyze Sentiment"):
    unseen_processed = preprocess_text(user_input)
    unseen_tokenized = tokenizer.texts_to_sequences([unseen_processed]) 
    unseen_padded = pad_sequences(unseen_tokenized, padding='post', maxlen=100)

    unseen_padded = tf.cast(unseen_padded, tf.float32)

    unseen_sentiments = model(unseen_padded) 

    sentiment_score = unseen_sentiments['dense_2']  
    sentiment_score = sentiment_score.numpy()[0][0]  
    
    sentiment = "Positive" if sentiment_score > 0.5 else "Negative"
    prediction_score = sentiment_score * 10  
    st.write(f"**Sentiment Score:** {prediction_score:.2f}")
    st.write(f"**Sentiment:** {sentiment}")
else:
    st.write("Please enter valid text.")
