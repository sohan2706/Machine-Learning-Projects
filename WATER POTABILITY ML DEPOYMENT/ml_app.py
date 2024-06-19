import streamlit as st
import pickle
import numpy as np

loaded_model = pickle.load(open('C:/Users/prasz/Documents/AI, ML, DL/WATER POTABILITY ML DEPOYMENT/trained_model.sav', 'rb'))

def predictor(input_data):
    
    input_data_array = np.array(input_data)
    
    input_data_reshaped = input_data_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
        return "The water is NOT potable"
    else:
        return "The water IS potable"


def main():
    
    st.title("Water Potability Prediction")
    
    ph = st.number_input(label = 'pH of Water(in pH units) ',step=1.,format="%.2f")
    hardness = st.number_input(label='Hardness(in mg/L)',step=1.,format="%.2f")
    solids = st.number_input(label='Solids(in ppm)',step=1.,format="%.2f")
    chloramines = st.number_input(label='Chloramines(in ppm)',step=1.,format="%.2f")
    sulfate = st.number_input(label='Sulafate(in mg/L)',step=1.,format="%.2f")
    conductivity = st.number_input(label='Conductivity(in μS/cm)',step=1.,format="%.2f")
    organic_carbon= st.number_input(label='Organic Carbon(in ppm)',step=1.,format="%.2f")
    trihalomethanes= st.number_input(label='Trihalomethanes(in μg/L)',step=1.,format="%.2f")
    turbidity = st.number_input(label='Turbidity(in NTU)',step=1.,format="%.2f")
    
    
    Potability = ''
    
    if st.button('Water Potability Test'):
        Potability = predictor([ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity])
        
        
    st.success(Potability)
    
    

if __name__ == '__main__':
    main()
    
    