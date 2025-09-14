# 🔥 Forest Fire Prediction App

This is an end-to-end Machine Learning web application built with Streamlit to predict the burned area of forest fires.

The application loads the [Forest Fires dataset](https://archive.ics.uci.edu/ml/datasets/Forest+Fires) from the UCI Machine Learning Repository, performs exploratory data analysis, trains a Random Forest Regressor, and provides an interactive interface for users to get predictions on new data.

## Features

- **Automated Data Loading**: Fetches data directly from the UCI repository URL.
- **Exploratory Data Analysis (EDA)**: Displays raw data, summary statistics, and visualizations like histograms and a correlation heatmap.
- **Interactive Prediction**: A user-friendly sidebar allows you to input meteorological conditions to predict the potential burned area.
- **ML Preprocessing**: Implements a robust pipeline including log transformation, one-hot encoding, and feature scaling.
- **Model Performance**: Shows key regression metrics (R² and MAE) for the trained model.

## 🛠️ Tech Stack

- **Python**
- **Streamlit** (for the web interface)
- **Pandas** (for data manipulation)
- **Scikit-learn** (for machine learning)
- **Seaborn & Matplotlib** (for data visualization)

## 🚀 How to Run Locally

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

- Python 3.8+ installed.
- `pip` and `venv` (or your preferred virtual environment tool).

### 2. Clone the Repository

Clone this project to your local machine.

```bash
git clone <your-repository-url>
cd forest-fire-app
```

### 3. Set Up a Virtual Environment

It's highly recommended to create a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit App

Launch the application using the Streamlit CLI.

```bash
streamlit run app.py
```

Your web browser should open a new tab with the running application at `http://localhost:8501`.

## ☁️ Deploying to Streamlit Community Cloud

Deploying this app is free and straightforward with Streamlit Community Cloud.

1.  **Push to GitHub**: Make sure your project (including `app.py`, `processing.py`, and `requirements.txt`) is in a public GitHub repository.
2.  **Sign Up**: Go to [share.streamlit.io](https://share.streamlit.io/) and sign up using your GitHub account.
3.  **Deploy**: Click the "New app" button, select your repository, specify the branch, and ensure the main file path is `app.py`. Then, click "Deploy!".

Streamlit will handle the rest, and your app will be live in a few minutes!