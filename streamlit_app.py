import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the pre-trained model
with open('language_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the corresponding CountVectorizer
with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_language(user_input):
    # Transform user input using the loaded CountVectorizer
    user_input_vec = vectorizer.transform([user_input])

    # Predict the language using the loaded model
    predicted_language = model.predict(user_input_vec)[0]

    # Get probability scores for each class
    probability_scores = model.predict_proba(user_input_vec)[0]

    return predicted_language, probability_scores

# Streamlit app
st.title("Language Detection Chat App")

# User input
user_input = st.text_input("Enter a text for language prediction:")

if user_input:
    # Get prediction and confidence scores
    predicted_language, confidence_scores = predict_language(user_input)

    # Display results
    st.write(f"Predicted Language: {predicted_language}")
    
    # Display confidence scores for each language
    st.write("Confidence Scores:")
    for lang, score in zip(model.classes_, confidence_scores):
        st.write(f"{lang}: {score:.2f}")
