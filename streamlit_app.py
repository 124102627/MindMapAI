import streamlit as st
import pandas as pd

st.title('Mind Map AI')
url = 'https://raw.githubusercontent.com/124102627/MindMapAI/master/standardised_mental_health_death_rates_2021.csv'
df = pd.read_csv(url)
