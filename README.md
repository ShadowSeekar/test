# 😃 Sentiment Analysis with LSTM — Streamlit App (test)
A simple and interactive Sentiment Analysis Web App built with Streamlit and TensorFlow LSTM model that classifies text input as Positive or Negative based on emotional tone.  
🔗 Live Demo:
https://senti-sum-jtrofqltttiihaqzlmp2ub.streamlit.app/

# 📖 Overview
This project is a sentiment classification web application that uses a Long Short-Term Memory (LSTM) neural network to estimate sentiment from user-provided text. It’s designed to be lightweight, easy to use, and deployable via Streamlit.  
The app takes text input, performs preprocessing to clean it, tokenizes and pads it using a saved tokenizer, and then feeds it to a TensorFlow model served via TensorFlow Serving using a TFSMLayer.

# ✨ Features
✔ Clean and interactive UI built with Streamlit  
✔ Preprocesses text to remove noise like HTML tags and stopwords  
✔ LSTM-based sentiment prediction in real time  
✔ Outputs sentiment score and label  
✔ Easy to deploy and extend  

# 🧠 How It Works
User Input:
User enters text in the Streamlit text area.    
Preprocessing:
Text is cleaned by removing HTML tags, non-alphabetic characters, and stopwords.  
Tokenization:
Preprocessed text is tokenized using a saved tokenizer.pkl.  
Padding:
Token sequences are padded to a fixed length for model compatibility.  
Inference:
Input is passed to the LSTM model via TFSMLayer and served using the TensorFlow SavedModel format.  
Output:
The app displays a sentiment score is in range [0-1] based on threshold.  

