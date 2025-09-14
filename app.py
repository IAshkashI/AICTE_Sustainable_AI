# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Import our custom processing functions
from processing import load_data, preprocess_data

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Forest Fire Prediction 🔥",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING AND PREPROCESSING ---
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'

# Load the raw data
raw_df = load_data(DATA_URL)

# Get the preprocessed data, scaler, and final columns
X_train, X_test, y_train, y_test, scaler, final_columns = preprocess_data(raw_df.copy())


# --- MODEL TRAINING ---
@st.cache_resource
def train_model(X_train, y_train):
    """Trains the RandomForestRegressor model and returns it."""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)


# --- UI: SIDEBAR FOR USER INPUT ---
st.sidebar.title("🌲 New Fire Prediction")
st.sidebar.header("Input Fire Conditions")

# Function to collect user input from the sidebar
def get_user_input():
    # Get unique sorted lists for selectboxes
    months = sorted(raw_df['month'].unique())
    days = sorted(raw_df['day'].unique())

    # Create input widgets
    month = st.sidebar.selectbox("Month", options=months, index=months.index('mar'))
    day = st.sidebar.selectbox("Day", options=days, index=days.index('fri'))
    temp = st.sidebar.slider("Temperature (°C)", float(raw_df['temp'].min()), float(raw_df['temp'].max()), 22.0)
    rh = st.sidebar.slider("Relative Humidity (%)", int(raw_df['RH'].min()), int(raw_df['RH'].max()), 50)
    wind = st.sidebar.slider("Wind Speed (km/h)", float(raw_df['wind'].min()), float(raw_df['wind'].max()), 4.0)
    ffmc = st.sidebar.number_input("FFMC", min_value=0.0, max_value=100.0, value=90.2)
    dmc = st.sidebar.number_input("DMC", min_value=0.0, max_value=300.0, value=60.5)
    dc = st.sidebar.number_input("DC", min_value=0.0, max_value=900.0, value=650.1)
    isi = st.sidebar.number_input("ISI", min_value=0.0, max_value=60.0, value=7.5)
    x = st.sidebar.number_input("X coordinate", min_value=1, max_value=9, value=7)
    y = st.sidebar.number_input("Y coordinate", min_value=1, max_value=9, value=5)

    # Store data in a dictionary
    user_data = {
        'X': x, 'Y': y, 'FFMC': ffmc, 'DMC': dmc, 'DC': dc, 'ISI': isi,
        'temp': temp, 'RH': rh, 'wind': wind, 'month': month, 'day': day
    }
    return pd.DataFrame(user_data, index=[0])

user_input_df = get_user_input()

# "Predict" button in the sidebar
predict_button = st.sidebar.button("Predict Burned Area")


# --- MAIN PAGE LAYOUT ---
st.title("🔥 Forest Fire Burned Area Prediction")
st.markdown("This app analyzes the Forest Fires dataset and predicts the total burned area based on meteorological conditions.")

# Create tabs for different sections of the app
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "ℹ️ About the Model"])

# --- TAB 1: EXPLORATORY DATA ANALYSIS ---
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    
    st.subheader("Raw Data Sample")
    st.dataframe(raw_df.head())

    st.subheader("Key Summary Statistics")
    st.write(raw_df.describe())

    st.subheader("Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of Temperature
        st.write("##### Temperature Distribution")
        fig, ax = plt.subplots()
        sns.histplot(raw_df['temp'], kde=True, ax=ax, color='orangered')
        ax.set_title("Distribution of Temperature")
        st.pyplot(fig)
        
    with col2:
        # Count of Fires per Month
        st.write("##### Fire Incidents per Month")
        fig, ax = plt.subplots()
        sns.countplot(data=raw_df, x='month', order=raw_df['month'].value_counts().index, ax=ax, palette='viridis')
        ax.set_title("Number of Fires per Month")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    # Correlation Heatmap
    st.write("##### Correlation Heatmap of Features")
    fig, ax = plt.subplots(figsize=(12, 8))
    # Select only numeric columns for correlation matrix
    numeric_df = raw_df.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix of Numerical Features")
    st.pyplot(fig)


# --- TAB 2: ABOUT THE MODEL ---
with tab2:
    st.header("About the Prediction Model")
    st.markdown("""
    The prediction is performed by a **Random Forest Regressor**, a powerful ensemble learning model. 
    Here's a brief overview of its performance on the unseen test data:
    """)
    
    # Make predictions on the test set
    y_pred_test = model.predict(X_test)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    
    # Note: Since the target was log-transformed, the MAE is on the log scale.
    # To make it more interpretable, we can inverse transform the predictions and true values.
    mae_original_scale = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_test))

    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    col1.metric("R-squared (R²)", f"{r2:.3f}")
    col2.metric("Mean Absolute Error (MAE)", f"{mae_original_scale:.3f} ha")
    
    st.markdown(f"""
    - **R-squared (R²)**: This value indicates that the model explains approximately **{r2:.1%}** of the variance in the test data. A higher value is better.
    - **Mean Absolute Error (MAE)**: On average, the model's predictions for the burned area are off by about **{mae_original_scale:.2f} hectares**. A lower value is better.
    """)
    
    st.info("💡 **Note:** The model was trained on the log-transformed `area` to handle the highly skewed distribution of the data. Performance metrics are calculated on the original scale (hectares) for better interpretability.")


# --- PREDICTION LOGIC AND DISPLAY ---
st.header("Prediction Result")
if predict_button:
    # 1. One-hot encode the user input
    input_processed = pd.get_dummies(user_input_df)
    
    # 2. Align columns with the training data
    # Add missing columns from the training set and fill with 0
    input_aligned = input_processed.reindex(columns=final_columns, fill_value=0)
    
    # 3. Scale the numerical features using the *fitted* scaler
    numerical_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']
    input_aligned[numerical_features] = scaler.transform(input_aligned[numerical_features])
    
    # 4. Make the prediction
    prediction_log = model.predict(input_aligned)
    
    # 5. Inverse transform the prediction to get the actual area
    prediction_hectares = np.expm1(prediction_log)[0]
    
    st.success(f"**Predicted Burned Area: `{prediction_hectares:.2f}` hectares**")
    
    st.markdown("---")
    st.write("#### Input Features Provided:")
    st.dataframe(user_input_df)
    
else:
    st.info("Please set the conditions in the sidebar and click 'Predict Burned Area'.")