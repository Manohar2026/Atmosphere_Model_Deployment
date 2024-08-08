#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the trained model
model_path = "random_forest_regression_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def main():
    st.sidebar.title("Menu")
    menu = st.sidebar.radio(
        "Select a page:",
        ("Introduction", "Prediction")
    )

    if menu == "Introduction":
        st.title("Introduction to Atmospheric Gases and PM2.5")
        st.write("""
        Air Quality Index (AQI):

        The Air Quality Index (AQI) is a numerical scale used to communicate the quality of the air in a specific location. It provides information about the level of pollution and its potential health effects. The AQI typically ranges from 0 to 500, with lower values indicating better air quality and higher values indicating poorer air quality.

        The AQI is calculated based on the concentrations of various air pollutants, including particulate matter (PM2.5 and PM10), ozone (O3), nitrogen dioxide (NO2), sulfur dioxide (SO2), and carbon monoxide (CO). These pollutants are measured using monitoring stations located in different areas.

        Each pollutant has its own AQI scale, and the overall AQI value is determined by the highest AQI value among these pollutants. The AQI is divided into several categories, each of which corresponds to a different level of health concern. These categories typically include "good," "moderate," "unhealthy for sensitive groups," "unhealthy," "very unhealthy," and "hazardous."

        PM2.5 Pollutant:

        PM2.5 refers to particulate matter that has a diameter of 2.5 micrometers or smaller. These particles are extremely small and can be inhaled deeply into the lungs, posing significant health risks. PM2.5 particles are generated from various sources, including vehicle emissions, industrial processes, construction activities, and burning of fossil fuels.

        Exposure to PM2.5 pollution has been linked to a wide range of health problems, including respiratory issues such as asthma, bronchitis, and lung cancer, as well as cardiovascular diseases and premature death. Due to their small size, PM2.5 particles can bypass the body's natural defenses and penetrate deep into the respiratory system, causing inflammation and damage to lung tissue.

        Monitoring PM2.5 levels is crucial for assessing air quality and protecting public health. High concentrations of PM2.5 pollution often occur in urban areas with heavy traffic and industrial activities, as well as during periods of wildfires or burning of biomass. Efforts to reduce PM2.5 pollution involve implementing measures to control emissions from vehicles and industries, promoting cleaner energy sources, and improving air quality monitoring and public awareness.
        """)

    elif menu == "Prediction":
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

        # Button to trigger prediction
        if st.button("Predict PM2.5"):
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
        if st.button("About"):
            st.write("Air Quality Index (AQI) Values, Levels of Health Concern:")
            st.write("0 to 50 - Good")
            st.write("51 to 100 - Moderate")
            st.write("101 to 150 - Unhealthy for Sensitive Groups")
            st.write("151 to 200 - Unhealthy")
            st.write("201 to 300 - Very Unhealthy")
            st.write("301 to 500 - Hazardous")

if __name__ == "__main__":
    main()
