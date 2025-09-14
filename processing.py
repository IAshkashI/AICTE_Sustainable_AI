# processing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Use Streamlit's caching to load data only once
import streamlit as st

@st.cache_data
def load_data(url):
    """
    Loads the Forest Fires dataset from a URL, maps month and day to numbers,
    and returns a pandas DataFrame.
    """
    df = pd.read_csv(url)
    # The dataset uses abbreviated month and day names. Let's map them for one-hot encoding.
    # This also helps in ordering if we were to treat them as ordinal.
    df['month'] = df['month'].str.lower()
    df['day'] = df['day'].str.lower()
    return df

@st.cache_data
def preprocess_data(df):
    """
    Performs the full preprocessing pipeline on the dataset.
    1. Applies log transformation to the 'area' column.
    2. Performs one-hot encoding on 'month' and 'day'.
    3. Splits data into training and testing sets.
    4. Scales numerical features using StandardScaler.
    
    Returns:
        - X_train_scaled, X_test_scaled, y_train, y_test
        - The fitted StandardScaler object
        - A list of the final columns after one-hot encoding
    """
    # 1. Log Transformation on the target variable 'area'
    # We use np.log1p which computes log(1+x) to handle zero values in 'area'.
    df['area'] = np.log1p(df['area'])

    # 2. One-Hot Encoding for categorical features
    # Create a copy to avoid modifying the original dataframe in-place
    processed_df = df.copy()
    processed_df = pd.get_dummies(processed_df, columns=['month', 'day'], drop_first=True)

    # Define features (X) and target (y)
    X = processed_df.drop('area', axis=1)
    y = processed_df['area']
    
    # Store the column order after one-hot encoding for later use in prediction
    final_columns = X.columns.tolist()

    # 3. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Feature Scaling
    # We identify numerical columns to scale. Exclude one-hot encoded columns.
    # Note: X and Y coordinates are already on a small scale, but we scale them for consistency.
    numerical_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']
    
    scaler = StandardScaler()
    
    # Fit the scaler ONLY on the training data's numerical features
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    
    # Transform the test data's numerical features using the SAME fitted scaler
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train, X_test, y_train, y_test, scaler, final_columns