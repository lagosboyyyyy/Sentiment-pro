import streamlit as st
import joblib

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.set_page_config(page_title="Sentiment Pro", page_icon="📈", layout="centered")

with st.sidebar:
    st.title("Model Info")
    st.info("This model is a Logistic Regression classifier trained on synthetic sentiment data.")
    st.markdown("---")
    st.write("Day 32 of DS > ML Journey")

st.title("Sentiment Analysis Pro 📈")
st.markdown("Enter text below to analyze its sentiment and see the model's confidence level.")

user_input = st.text_area("Your Text:", placeholder="Type something like 'This is an amazing day'...", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        
        probs = model.predict_proba(input_vector)[0]
        confidence = probs[prediction] * 100
        
        st.markdown("### Result:")
        col1, col2 = st.columns(2)
        
        sentiment_label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        color = "green" if prediction == 1 else "red"
        
        with col1:
            st.metric(label="Sentiment", value=sentiment_label)
        
        with col2:
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            
        if prediction == 1:
            st.success(f"The model is {confidence:.1f}% sure this is positive.")
            st.progress(int(confidence))
        else:
            st.error(f"The model is {confidence:.1f}% sure this is negative.")
            st.progress(int(confidence))
