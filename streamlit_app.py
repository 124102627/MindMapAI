import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
st.title("MindMap AI – Mental Health Risk Forecasting")
st.markdown("Predicting country-level mental health risks using Random Forest and XGBoost.")

df = pd.read_csv("cleaned_mental_health_data.csv")

# Sidebar filter
country_filter = st.sidebar.multiselect("Select Countries", df["Country"].unique(), default=df["Country"].unique())

# Filtered data
filtered_df = df[df["Country"].isin(country_filter)]

# Feature & target split
X = filtered_df.drop(columns=["Country", "Self Harm Rate"])
y = filtered_df["Self Harm Rate"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predictions
rf_preds = rf.predict(X_test)
xgb_preds = xgb.predict(X_test)

# Model Evaluation
st.subheader("Model Performance")
col1, col2 = st.columns(2)
col1.metric("Random Forest R²", f"{r2_score(y_test, rf_preds):.2f}")
col1.metric("Random Forest RMSE", f"{mean_squared_error(y_test, rf_preds, squared=False):.2f}")

col2.metric("XGBoost R²", f"{r2_score(y_test, xgb_preds):.2f}")
col2.metric("XGBoost RMSE", f"{mean_squared_error(y_test, xgb_preds, squared=False):.2f}")

# Predict full set with Random Forest
df["Predicted Self Harm Rate"] = rf.predict(df.drop(columns=["Country", "Self Harm Rate"]))

# Show Top 5 countries at risk
st.subheader("Top 5 Predicted High-Risk Countries (Self Harm Rate)")
top5 = df.sort_values(by="Predicted Self Harm Rate", ascending=False).head(5)
st.dataframe(top5[["Country", "Predicted Self Harm Rate"]])

# Plot Predictions
st.subheader("Actual vs Predicted – Random Forest")
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=rf_preds)
plt.xlabel("Actual Self Harm Rate")
plt.ylabel("Predicted Self Harm Rate")
plt.title("Random Forest Prediction Accuracy")
st.pyplot(plt)

# Feature Importance
st.subheader("Feature Importance – Random Forest")
importance = pd.Series(rf.feature_importances_, index=X.columns)
st.bar_chart(importance.sort_values(ascending=True))

# About
st.sidebar.markdown("Created by Group 28R – MindMap AI")

