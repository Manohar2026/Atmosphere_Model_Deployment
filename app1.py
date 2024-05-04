import streamlit as st
import scikit-learn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the trained model
model = RandomForestRegressor()  # Instantiate the model object

# Assuming you have trained the model previously and saved it as "random_forest_regression_model.joblib"
model_path = "random_forest_regression_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def main():
    st.title("PM2.5 Prediction App")

    # Create input fields for each feature
    t = st.slider("Average Temperature (°C)", -20.0, 40.0, 0.0)
    tm = st.slider("Minimum Temperature (°C)", -20.0, 40.0, 0.0)
    tm = st.slider("Maximum Temperature (°C)", -20.0, 40.0, 0.0)
    slp = st.slider("Atmospheric Pressure (hPa)", 900.0, 1100.0, 1013.0)
    h = st.slider("Relative Humidity (%)", 0, 100, 50)
    vv = st.slider("Visibility (km)", 0.0, 50.0, 10.0)
    v = st.slider("Wind Speed (km/hr)", 0.0, 100.0, 10.0)
    vm = st.slider("Max Wind Speed (km/hr)", 0.0, 200.0, 20.0)

    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        "T": [t],
        "TM": [tm],
        "Tm": [tm],
        "SLP": [slp],
        "H": [h],
        "VV": [vv],
        "V": [v],
        "VM": [vm]
    })

    # Predict PM2.5 value
    prediction = model.predict(input_data)[0]

    st.write(f"Predicted PM2.5 value: {prediction} µg/m³")

if __name__ == "__main__":
    main()
    
