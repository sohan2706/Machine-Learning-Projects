import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

def show_predict_page():
    st.title("Water Potability Prediction")
    
    st.write("""### We need some data to predict whether the water is potable or not""")
    
