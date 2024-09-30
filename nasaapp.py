import streamlit as st
import pandas as pd

# Display a greeting
st.write("Hello, World!")

# Read the CSV file with error handling
 = pd.read_csv('PlanetarySystems_data.csv', on_bad_lines='skip', engine='python')

# Display the dataframe
st.write(df)
