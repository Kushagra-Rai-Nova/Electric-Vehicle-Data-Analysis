import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Title
st.title('Electric Vehicle Data Analysis')

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Read CSV file
    ev_data = pd.read_csv(uploaded_file)

    # Show data info
    st.subheader('Data Info:')
    st.write(ev_data.info())

    # Data cleaning (drop NA values)
    ev_data.dropna(inplace=True)

    # Show cleaned data
    st.subheader('Cleaned Data:')
    st.write(ev_data)

    # Visualizations
    st.subheader('Electric Vehicle Adoption Over Time:')
    plt.figure(figsize=(12, 6))
    ev_adoption_by_year = ev_data['Model Year'].value_counts().sort_index()
    sns.barplot(x=ev_adoption_by_year.index, y=ev_adoption_by_year.values, palette="viridis")
    plt.xlabel('Model Year')
    plt.ylabel('Number of Vehicles Registered')
    plt.xticks(rotation=45)
    st.pyplot()

    # Top 3 counties by EV registrations
    ev_county_distribution = ev_data['County'].value_counts()
    top_counties = ev_county_distribution.head(3).index

    # Filter data for top counties
    top_counties_data = ev_data[ev_data['County'].isin(top_counties)]

    # Group by County and City, count vehicles, and sort
    ev_city_distribution_top_counties = top_counties_data.groupby(['County', 'City']).size().sort_values(ascending=False).reset_index(name='Number of Vehicles')

    # Get top 10 cities
    top_cities = ev_city_distribution_top_counties.head(10)

    # Create bar chart for top cities in top counties
    st.subheader('Top Cities in Top Counties by EV Registrations')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Number of Vehicles', y='City', hue='County', data=top_cities, palette="magma", dodge=False, ax=ax)
    ax.set_title('Top Cities in Top Counties by EV Registrations')
    ax.set_xlabel('Number of Vehicles Registered')
    ax.set_ylabel('City')
    ax.legend(title='County')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

    # Top 10 Popular EV Makes
    st.subheader('Top 10 Popular EV Makes:')
    ev_make_distribution = ev_data['Make'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=ev_make_distribution.values, y=ev_make_distribution.index, palette="cubehelix")
    plt.xlabel('Number of Vehicles Registered')
    plt.ylabel('Make')
    st.pyplot()

    # Distribution of Electric Vehicle Types
    st.subheader('Distribution of Electric Vehicle Types:')
    ev_type_distribution = ev_data['Electric Vehicle Type'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=ev_type_distribution.values, y=ev_type_distribution.index, palette="rocket")
    plt.xlabel('Number of Vehicles Registered')
    plt.ylabel('Electric Vehicle Type')
    st.pyplot()

    # Forecasting
    st.subheader('Forecasting EV Registrations:')
    ev_registration_counts = ev_data['Model Year'].value_counts().sort_index()

    def exp_growth(x, a, b):
        return a * np.exp(b * x)

    filtered_years = ev_registration_counts[ev_registration_counts.index <= 2023]
    x_data = filtered_years.index - filtered_years.index.min()
    y_data = filtered_years.values
    params, _ = curve_fit(exp_growth, x_data, y_data)

    forecast_years = np.arange(2024, 2030) - filtered_years.index.min()
    forecasted_values = exp_growth(forecast_years, *params)

    forecasted_df = pd.DataFrame({
        'Year': np.arange(2024, 2030),
        'Forecasted Registrations': forecasted_values.astype(int)
    })
    st.write(forecasted_df)

    # Plot actual vs forecasted registrations
    plt.figure(figsize=(12, 8))
    plt.plot(filtered_years.index, filtered_years.values, 'bo-', label='Actual Registrations')
    plt.plot(np.arange(2024, 2030), forecasted_values, 'ro--', label='Forecasted Registrations')
    plt.title('Actual vs Forecasted EV Registrations')
    plt.xlabel('Year')
    plt.ylabel('Number of EV Registrations')
    plt.legend()
    st.pyplot()

    # Distribution of Electric Vehicle Ranges
    st.subheader('Distribution of Electric Vehicle Ranges')
    plt.figure(figsize=(12, 6))
    sns.histplot(ev_data['Electric Range'], bins=30, kde=True, color='royalblue')
    plt.title('Distribution of Electric Vehicle Ranges')
    plt.xlabel('Electric Range (miles)')
    plt.ylabel('Number of Vehicles')
    plt.axvline(ev_data['Electric Range'].mean(), color='red', linestyle='--', label=f'Mean Range: {ev_data["Electric Range"].mean():.2f} miles')
    plt.legend()
    st.pyplot()

    st.subheader('Top 10 Models by Average Electric Range in Top Makes')
    average_range_by_model = ev_data.groupby(['Make', 'Model'])['Electric Range'].mean().sort_values(ascending=False).reset_index()

    # the top 10 models with the highest average electric range
    top_range_models = average_range_by_model.head(10)

    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Electric Range', y='Model', hue='Make', data=top_range_models, palette="cool",dodge=False)
    plt.title('Top 10 Models by Average Electric Range in Top Makes')
    plt.xlabel('Average Electric Range (miles)')    
    plt.ylabel('Model')
    plt.legend(title='Make', loc='center right')
    st.pyplot()

    st.subheader('Average Electric Range by Model Year')
    # calculating the average electric range by model year
    average_range_by_year = ev_data.groupby('Model Year')['Electric Range'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Model Year', y='Electric Range', data=average_range_by_year, marker='o', color='lightgreen')
    plt.title('Average Electric Range by Model Year')
    plt.xlabel('Model Year')
    plt.ylabel('Average Electric Range (miles)')
    plt.grid(True)
    st.pyplot()

