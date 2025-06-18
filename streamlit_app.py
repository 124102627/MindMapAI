import streamlit as st
import pandas as pd

st.title('Mind Map AI')

df = pd.read_csv('https://github.com/124102627/MindMapAI/blob/master/standardised_mental_health_death_rates_2021.csv.xlsx')
df
