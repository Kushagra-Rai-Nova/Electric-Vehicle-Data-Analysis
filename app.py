# ev_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

# Load data
ev_data = pd.read_csv('Electric_Vehicle_Population_Data.csv')

# Title
st.title('Electric Vehicle Dashboard')

# Sidebar for data exploration options
st.sidebar.title('Explore Data')
explore_option = st.sidebar.selectbox('Select an option', ['EV Adoption Over Time', 'Geographical Distribution', 'Electric Vehicle Types', 'EV Manufacturers'])

# Filter out missing values
ev_data = ev_data.dropna()

if explore_option == 'EV Adoption Over Time':
    st.subheader('EV Adoption Over Time')
    ev_adoption_by_year = ev_data['Model Year'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=ev_adoption_by_year.index, y=ev_adoption_by_year.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.xlabel('Model Year')
    plt.ylabel('Number of Vehicles Registered')
    st.pyplot()

elif explore_option == 'Geographical Distribution':
    st.subheader('Geographical Distribution')
    ev_county_distribution = ev_data['County'].value_counts()
    top_counties = ev_county_distribution.head(3).index
    top_counties_data = ev_data[ev_data['County'].isin(top_counties)]
    ev_city_distribution_top_counties = top_counties_data.groupby(['County', 'City']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')
    top_cities = ev_city_distribution_top_counties.head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Number of Vehicles', y='City', hue='County', data=top_cities, palette="magma")
    plt.xlabel('Number of Vehicles Registered')
    plt.ylabel('City')
    plt.legend(title='County')
    plt.tight_layout()
    st.pyplot()

elif explore_option == 'Electric Vehicle Types':
    st.subheader('Electric Vehicle Types')
    ev_type_distribution = ev_data['Electric Vehicle Type'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=ev_type_distribution.values, y=ev_type_distribution.index, palette="rocket")
    plt.xlabel('Number of Vehicles Registered')
    plt.ylabel('Electric Vehicle Type')
    plt.tight_layout()
    st.pyplot()

elif explore_option == 'EV Manufacturers':
    st.subheader('EV Manufacturers')
    ev_make_distribution = ev_data['Make'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=ev_make_distribution.values, y=ev_make_distribution.index, palette="cubehelix")
    plt.xlabel('Number of Vehicles Registered')
    plt.ylabel('Make')
    plt.tight_layout()
    st.pyplot()

# Additional visualizations and forecasting can be added similarly based on user selection
