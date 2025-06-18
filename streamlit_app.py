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

st.title("🔮 Country-Level Mental Health Death Rate Prediction")

df, model, scaler = load_and_train_model()

st.subheader("Top 10 Countries with Highest Predicted Death Rates")
top10 = df[['Country', 'Predicted_Death_Rate']].sort_values(by='Predicted_Death_Rate', ascending=False).head(10)
st.dataframe(top10)

st.subheader("View All Predictions")
st.dataframe(df[['Country', 'Predicted_Death_Rate', 'Total_Death_Rate']].sort_values(by='Predicted_Death_Rate', ascending=False))

import matplotlib.pyplot as plt

country = st.selectbox("Choose a Country", df['Country'].unique())
selected = df[df['Country'] == country]

st.metric("Predicted Death Rate", f"{selected['Predicted_Death_Rate'].values[0]:.2f}")
st.metric("Reported Death Rate", f"{selected['Total_Death_Rate'].values[0]:.2f}")

# Visual: Breakdown by disorder
features = selected[[
    'Dementia (<65)', 'Dementia (65+)',
    'Alcohol disorders (<65)', 'Alcohol disorders (65+)',
    'Drug dependence (<65)', 'Drug dependence (65+)',
]].T
features.columns = ['Death Rate']

fig, ax = plt.subplots(figsize=(8, 5))
features.plot(kind='bar', ax=ax, legend=False, color='coral')
ax.set_title(f"Mental Health Death Rate Breakdown – {country}")
ax.set_ylabel("Death Rate per 100k")
ax.grid(axis='y')
st.pyplot(fig)

