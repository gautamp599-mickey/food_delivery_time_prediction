import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(page_title="Delivery Time Predictor", page_icon="🛵")

# 1. Load the saved assets (Cache them so they only load once)
@st.cache_resource
def load_assets():
    preprocessor = joblib.load('preprocessor.pkl')
    model = load_model('delivery_nn_model.keras')
    return preprocessor, model

preprocessor, model = load_assets()

st.title("🛵 Food Delivery ETA Predictor")
st.markdown("Enter the order and traffic details below to generate a neural network prediction.")

# 2. Create UI inputs for the user
col1, col2 = st.columns(2)

with col1:
    st.subheader("Order Details")
    cuisine = st.selectbox("Cuisine Type", ['South Indian', 'North Indian', 'Pizza', 'Biryani', 'Burger', 'Chinese', 'Desserts', 'Bakery', 'Cafe'])
    order_items = st.number_input("Number of Items", min_value=1, max_value=20, value=3)
    prep_time = st.slider("Estimated Prep Time (Mins)", 5, 60, 20)
    priority = st.selectbox("Delivery Priority", ['Normal', 'Priority', 'VIP'])
    rest_load = st.selectbox("Restaurant Load", ['Low', 'Medium', 'High'])
    
    st.subheader("Locations & Ratings")
    pickup = st.selectbox("Pickup Zone", ['Residential', 'Commercial', 'CBD', 'Industrial', 'Suburban'])
    dropoff = st.selectbox("Dropoff Zone", ['Residential', 'Commercial', 'CBD', 'Industrial', 'Suburban'])
    rider_exp = st.number_input("Rider Experience (Years)", 0.0, 15.0, 2.5)
    rider_rate = st.slider("Rider Rating", 1.0, 5.0, 4.5)
    rest_rate = st.slider("Restaurant Rating", 1.0, 5.0, 4.2)

with col2:
    st.subheader("Logistics & Environment")
    weather = st.selectbox("Weather", ['Clear', 'Cloudy', 'Rain', 'Storm', 'Fog'], index=2)
    traffic = st.selectbox("Traffic Level", ['Low', 'Moderate', 'High', 'Severe'])
    vehicle = st.selectbox("Vehicle Type", ['Scooter', 'Bike', 'Electric Scooter', 'Bicycle'])
    road_dist = st.number_input("Road Distance (km)", min_value=0.5, max_value=30.0, value=5.5)
    dist_cat = st.selectbox("Distance Category", ['Short', 'Medium', 'Long'], index=1)
    avg_speed = st.number_input("Average Speed (kmph)", min_value=10.0, max_value=60.0, value=35.0)
    signals = st.slider("Number of Signals", 0, 20, 5)
    
    
    st.subheader("Time & Date")
    order_hour = st.slider("Order Hour (24h)", 0, 23, 19)
    day = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
    is_festival = st.radio("Is it a Festival?", [0, 1], index=0)

# 3. Reconstruct the input into a DataFrame matching your training data exactly
if st.button("Predict Delivery Time", type="primary"):
    
    # Create a dictionary matching the exact column names of df.iloc[:,:-1]
    input_data = {
        'Order_Hour': [order_hour],
        'Day_of_Week': [day],
        'Is_Weekend': [is_weekend],
        'Is_Festival': [is_festival],
        'Weather': [weather],
        'Pickup_Zone': [pickup],
        'Dropoff_Zone': [dropoff],
        'Vehicle_Type': [vehicle],
        'Rider_Experience_Years': [rider_exp],
        'Rider_Rating': [rider_rate],
        'Restaurant_Rating': [rest_rate],
        'Cuisine_Type': [cuisine],
        'Order_Items': [order_items],
        'Restaurant_Load': [rest_load],
        'Preparation_Time_Min': [prep_time],
        'Road_Distance_km': [road_dist],
        'Delivery_Distance_Category': [dist_cat],
        'Traffic_Level': [traffic],
        'Number_of_Signals': [signals],
        'Average_Speed_kmph': [avg_speed],
        'Delivery_Priority': [priority]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # 4. Transform and Predict
    try:
        # Preprocess the data
        processed_input = preprocessor.transform(input_df)
        
        # Make Prediction
        prediction = model.predict(processed_input, verbose=0)[0][0]
        
        # Display Result
        st.success(f"### Estimated Delivery Time: {int(np.round(prediction))} Minutes")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")