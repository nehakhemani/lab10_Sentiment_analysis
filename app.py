import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor

def run():
    model = joblib.load('model.joblib') 

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    predicted_sentiment = ""
    # Prediction action on button click
    if st.button("Predict"):
        # Preprocess the input using the preprocessor
        preprocessed_text = pd.Series([userinput])  # Convert input to Series for preprocessing
        processed_text = preprocessor().fit_transform(preprocessed_text)  # Apply preprocessing steps
        
        # Predict sentiment using the model
        predicted_sentiment = model.predict(processed_text)[0]  # Get the first prediction
        
        # Interpret the result
        if predicted_sentiment == 1:
            output = 'positive 👍'
        else:
            output = 'negative 👎'
        
        # Display the sentiment result
        sentiment = f'Predicted sentiment of "{userinput}" is {output}.'
        st.success(sentiment)

if __name__ == "__main__":
    run()