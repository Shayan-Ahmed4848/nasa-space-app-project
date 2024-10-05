# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set the title of the Streamlit app
st.title('Exoplanet Habitability Predictor 🌍✨')

# Cache the data loading functions
@st.cache_data
def load_data():
    planetary_data = pd.read_csv('PlanetarySystems_data.csv')
    stellar_data = pd.read_csv('StellarHosts_data.csv')
    transiting_data = pd.read_csv('TransitingPlanetsTable_data.csv')
    return planetary_data, stellar_data, transiting_data

# Load datasets
planetary_data, stellar_data, transiting_data = load_data()

# Display raw data for users if needed
st.subheader('Raw Data: Planetary Systems')
st.write(planetary_data.head())

# Data Preprocessing
def preprocess_data(planetary_data):
    # Dropping null values for simplicity
    planetary_data = planetary_data.dropna(subset=['pl_orbper', 'pl_rade', 'pl_orbeccen', 'pl_eqt', 'st_mass', 'st_teff'])
    
    # Feature Selection
    features = planetary_data[['pl_orbper', 'pl_rade', 'pl_orbeccen', 'pl_eqt', 'st_mass', 'st_teff']]
    labels = planetary_data['pl_habitable']
    
    return features, labels

features, labels = preprocess_data(planetary_data)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Display accuracy score
accuracy = accuracy_score(y_test, y_pred)
st.subheader(f'Model Accuracy: {accuracy*100:.2f}%')

# Function for user input-based habitability prediction
def user_input():
    pl_orbper = st.sidebar.slider('Orbital Period (days)', float(planetary_data['pl_orbper'].min()), float(planetary_data['pl_orbper'].max()), float(planetary_data['pl_orbper'].mean()))
    pl_rade = st.sidebar.slider('Planet Radius (Earth Radii)', float(planetary_data['pl_rade'].min()), float(planetary_data['pl_rade'].max()), float(planetary_data['pl_rade'].mean()))
    pl_orbeccen = st.sidebar.slider('Orbital Eccentricity', float(planetary_data['pl_orbeccen'].min()), float(planetary_data['pl_orbeccen'].max()), float(planetary_data['pl_orbeccen'].mean()))
    pl_eqt = st.sidebar.slider('Equilibrium Temperature (K)', float(planetary_data['pl_eqt'].min()), float(planetary_data['pl_eqt'].max()), float(planetary_data['pl_eqt'].mean()))
    st_mass = st.sidebar.slider('Host Star Mass (Solar Masses)', float(planetary_data['st_mass'].min()), float(planetary_data['st_mass'].max()), float(planetary_data['st_mass'].mean()))
    st_teff = st.sidebar.slider('Host Star Effective Temperature (K)', float(planetary_data['st_teff'].min()), float(planetary_data['st_teff'].max()), float(planetary_data['st_teff'].mean()))
    
    # Store inputs in dictionary
    user_data = {
        'pl_orbper': pl_orbper,
        'pl_rade': pl_rade,
        'pl_orbeccen': pl_orbeccen,
        'pl_eqt': pl_eqt,
        'st_mass': st_mass,
        'st_teff': st_teff
    }
    
    # Convert to DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

# User input
input_data = user_input()

# Show the user's input data in the main page
st.subheader('User Input Parameters')
st.write(input_data)

# Make predictions on the user's input data
prediction = model.predict(input_data)

# Display the prediction result
st.subheader('Habitability Prediction')
if prediction[0] == 1:
    st.success('This exoplanet is likely habitable!')
else:
    st.warning('This exoplanet is not likely habitable.')

# Visualization - Feature Importance
st.subheader('Feature Importance')
importances = pd.Series(model.feature_importances_, index=features.columns)
importances.plot(kind='barh')
st.pyplot(plt)

# Conclusion
st.subheader('Project Overview')
st.markdown("""
**Exoplanet Habitability Predictor** is a machine learning-powered application designed to analyze various planetary features and predict whether a given exoplanet is potentially habitable. 
The prediction is based on features such as orbital period, radius, eccentricity, equilibrium temperature, host star mass, and effective temperature.
""")
