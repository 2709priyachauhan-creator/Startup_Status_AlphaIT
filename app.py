import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#Page configuration
st.set_page_config(page_title="Startup Analysis Dashboard", layout="wide")

st.title("Startup Success & Funding Dashboard")

#Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a View:",["EDA Dashboard","Predict Success"])

# --- VIEW 1: DASHBOARD ---
if page == "EDA Dashboard":
    st.header(" Dynamic Startup Analysis")
    st.write("Filter and explore the imbalanced startup dataset interactively.")
    
    try:
        # Load dataset
        df = pd.read_csv(r'C:\Users\PRIYA\OneDrive\Desktop\Priya Chauhan\Alpa IT\Startup_imbalanced_dataset_task\investments_VC.csv', encoding='latin1')
        df.columns = df.columns.str.strip()
        
        # Clean the funding column so we can plot it mathematically 
        df['funding_total_usd'] = df['funding_total_usd'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')
        
        # 1. DYNAMIC SIDEBAR FILTERS
        st.sidebar.markdown("---")
        st.sidebar.subheader("Dashboard Filters")
        
        available_statuses = df['status'].dropna().unique()
        selected_status = st.sidebar.multiselect(
            "Filter by Startup Status", 
            options=available_statuses, 
            default=available_statuses # Selects all by default
        )
        
        # Apply the filter to the dataframe
        filtered_df = df[df['status'].isin(selected_status)]
        
        # 2. LIVE METRIC CARDS
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Startups (Filtered)", f"{len(filtered_df):,}")
        
        # Check if there is data to calculate mean, otherwise show 0
        avg_funding = filtered_df['funding_total_usd'].mean()
        col2.metric("Average Funding", f"${avg_funding:,.0f}" if pd.notnull(avg_funding) else "$0")
        
        max_rounds = filtered_df['funding_rounds'].max()
        col3.metric("Max Funding Rounds", int(max_rounds) if pd.notnull(max_rounds) else 0)
        
        st.markdown("---")
        
        # 3. INTERACTIVE DATA & CHARTS
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Live Data Snapshot")
            # Streamlit's native dataframe allows sorting and scrolling
            st.dataframe(filtered_df[['name', 'market', 'funding_total_usd', 'funding_rounds', 'status']], height=350)
            
        with col_right:
            st.subheader("Filtered Class Imbalance")
            # Streamlit's native interactive bar chart
            status_counts = filtered_df['status'].value_counts()
            st.bar_chart(status_counts, color="#1f77b4")
            
        # 4. INTERACTIVE SCATTER PLOT
        st.subheader("Funding Amount vs. Funding Rounds")
        st.write("Hover your mouse over the points to see details!")
        
        # Drop NaNs just for the chart so it renders cleanly
        chart_data = filtered_df.dropna(subset=['funding_total_usd', 'funding_rounds'])
        st.scatter_chart(
            data=chart_data, 
            x='funding_rounds', 
            y='funding_total_usd', 
            color='status',
            height=400
        )
            
    except FileNotFoundError:
        st.error("Make sure 'investments_VC.csv' is in the exact folder specified!")

#---VIEW 2: PREDICTION TOOL---
elif page == "Predict Success":
    st.header("Random Forest Predictor")
    st.write("Enter the funding details below to predict the startup's status.")

    #Input fields
    funding_total = st.number_input("Total Funding (USD)", min_value=0, value=1500000)
    funding_rounds = st.number_input("Second Funding (USD)", min_value=1, value=2)
    seed_funding = st.number_input("Seed Funding (USD)", min_value=0, value=500000)

    if st.button("Run Prediction"):
        try:
            #Load the saved models
            rf_model = joblib.load('rf_model.pkl')
            saved_scaler = joblib.load('scaler.pkl')

            #Create an array of 23 zeroes to match the X_numeric shape from the notebook
            input_features = np.zeros((1,24))

            #Map our 3 inputs to their approximate column indices from the dataset
            input_features[0,0] = funding_total #funding_total_usd
            input_features[0,1] = funding_rounds #funding_rounds
            input_features[0,3] = seed_funding #seed

            #Scale and predict
            scaled_features = saved_scaler.transform(input_features)
            prediction = rf_model.predict(scaled_features)

            st.success(f"**Model Prediction:** This startup is classified as '{prediction[0]}'")
        
        except Exception as e:
            st.error(f"Error running prediction. Make sure your .pkl files are in the folder! Details: {e}")