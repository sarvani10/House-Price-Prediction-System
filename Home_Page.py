import streamlit as st


st.set_page_config(page_title="Home", layout="centered")

st.title("House Price Predictor")
st.write("Welcome! Use the left sidebar to navigate:")

st.markdown("""
### 🔹 **Predict Price**
Enter inputs and get predictions using:
- Linear Regression (Scratch)
- Decision Tree (Scratch)

### 🔹 **Model Comparison**
See R² scores and metrics on a separate page.

""")

st.success("Go to **Prediction** page from the left sidebar!")
