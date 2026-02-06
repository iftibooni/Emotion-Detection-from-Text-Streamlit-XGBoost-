import joblib
import streamlit as st

# Load the saved model, vectorizer, and label encoder
xgb_model = joblib.load('Xg_Boost_model.pkl')

vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')


# Inject custom CSS for background and styling
st.markdown("""
    <style>
    /* Add background color */
    body {
        background-color: #f0f2f6; /* Light pastel blue background */
    }

    /* Style the text area */
    .stTextArea textarea {
        background-color: #39345b;
        border: 2px solid #007BFF; /* Blue border */
        border-radius: 10px;
        font-size: 18px;
        color: #FF0000; /* Changed text color to red */
    }

    /* Style the main button */
    .stButton button {
        background-color: #007BFF;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }

    /* Add hover effect to the button */
    .stButton button:hover {
        background-color: #0056b3;
        cursor: pointer;
    }

    /* Center the content */
    .block-container {
        padding: 2rem 3rem;
        text-align: center;
    }

    /* Style the title */
    h1 {
        color: #007BFF;
        font-family: 'Arial Black', sans-serif;
        font-size: 48px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }

    /* Style the detected emotion */
    .emotion {
        font-size: 36px;
        color: #ff6347; /* Tomato color */
        font-family: 'Georgia', serif;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to predict emotion
def predict_emotion(text):
    # Transform text using TF-IDF vectorizer
    text_transformed = vectorizer.transform([text])

    # Make prediction using the XGBoost model
    prediction = xgb_model.predict(text_transformed)

    # Inverse transform to get the actual emotion label
    emotion = label_encoder.inverse_transform(prediction)

    return emotion[0] if emotion else "Unknown Emotion"

# Streamlit Page Title
st.title("Emotion Detection from Text")

# Instruction Text
st.write("Enter a text to analyze the emotion:")

# Input Text Area
user_input = st.text_area("Type your text here", placeholder="Enter your text...")

# Button to Predict Emotion
if st.button("Analyze Emotion"):
    if user_input:
        # Predict emotion
        emotion = predict_emotion(user_input)
        st.markdown(f"<p class='emotion'>The detected emotion is: <strong>{emotion}</strong></p>", unsafe_allow_html=True)
    else:
        st.write("Please enter some text to analyze.")

# Footer
st.write("---")
st.write("Emotion Detection App")

# Command-line part for testing (Optional for script testing)
if __name__ == "__main__":
    texts = input("Enter Text Here: ")
    texts = [texts]
    texts_vectorized = vectorizer.transform(texts)
    predictions = xgb_model.predict(texts_vectorized)
    predicted_emotions = label_encoder.inverse_transform(predictions)

    print(f'Text: "{texts[0]}" -> Predicted Emotion: {predicted_emotions[0]}')