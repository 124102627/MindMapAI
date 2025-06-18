import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st

st.set_page_config(page_title="MindMap AI", layout="wide")

@st.cache_data
def load_and_train_model():
    url = "https://raw.githubusercontent.com/124102627/MindMapAI/master/standardised_mental_health_death_rates_2021.csv"
    df = pd.read_csv(url)

    feature_cols = [
        'Dementia (<65)', 'Dementia (65+)',
        'Alcohol disorders (<65)', 'Alcohol disorders (65+)',
        'Drug dependence (<65)', 'Drug dependence (65+)'
    ]
    df['Total_Death_Rate'] = df[feature_cols].sum(axis=1)

    X = df[feature_cols]
    y = df['Total_Death_Rate']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    df['Predicted_Death_Rate'] = model.predict(X_scaled)
    return df, model, scaler
