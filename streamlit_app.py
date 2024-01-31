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

# Function to predict language
def predict_language(user_input):
    # Transform user input using the loaded CountVectorizer
    user_input_vec = vectorizer.transform([user_input])

    # Predict the language using the loaded model
    predicted_language = model.predict(user_input_vec)[0]

    # Get probability scores for each class
    probability_scores = model.predict_proba(user_input_vec)[0]

    return predicted_language, probability_scores

# Function to track and update chat history
def update_chat_history(user_input, predicted_language, confidence_score):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({
        'user_input': user_input,
        'predicted_language': predicted_language,
        'confidence_score': confidence_score
    })

# Streamlit app
st.title("Language Detection Chat Interface")

# Display creator's name in small fonts
st.markdown("<sub>Creator: Md Khalid Siddiqui</sub>", unsafe_allow_html=True)

# Track and display visitor count
visitor_count = track_visitor_count()
st.write(f"Visitor Count: {visitor_count}")


# Display chat history
if 'chat_history' in st.session_state:
    st.write("Chat History:")
    for chat in st.session_state.chat_history:
        st.write(f"{chat['user_input']} (User)")
        st.write(f"Predicted Language: {chat['predicted_language']}")
        st.write(f"Confidence Score: {chat['confidence_score']:.2f}")
        st.write("----")

# User input
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Get prediction and confidence scores
    predicted_language, confidence_scores = predict_language(user_input)

    # Get the confidence score for the predicted language
    confidence_score = confidence_scores[model.classes_ == predicted_language][0]

    # Update and display chat history
    update_chat_history(user_input, predicted_language, confidence_score)

    # Display model's response in the chat area
    st.write(f"Model: Predicted Language: {predicted_language}")
    st.write(f"Model: Confidence Score: {confidence_score:.2f}")
