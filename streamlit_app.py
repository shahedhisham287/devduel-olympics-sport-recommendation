import os
import sys
os.path.dirname(sys.executable)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import google.generativeai as genai

# Load and preprocess the data
df_sports = pd.read_csv(r'C:\Users\shahe\Downloads\sport.csv')
df_countries = pd.read_csv(r'C:\Users\shahe\Downloads\country.csv')

# Prepare the features and target for the model
sports = df_sports['sport'].unique()
countries = df_countries['country'].unique()

sport_encoder = {sport: idx for idx, sport in enumerate(sports)}
country_encoder = {country: idx for idx, country in enumerate(countries)}

df_sports['sport_encoded'] = df_sports['sport'].map(sport_encoder)
df_sports['country_encoded'] = df_sports['country_with_most_gold_medals'].map(country_encoder)

X = pd.get_dummies(df_sports[['sport_encoded']], columns=['sport_encoded'])
y = df_sports['country_encoded']

# Define and train the Logistic Regression model
clf = LogisticRegression(max_iter=200, random_state=42)
clf.fit(X, y)

def get_best_country(sport):
    sport_encoded = sport_encoder.get(sport)
    if sport_encoded is None:
        return "Sport not recognized", None, None

    user_vector = pd.get_dummies(pd.DataFrame({'sport_encoded': [sport_encoded]}), columns=['sport_encoded'])
    user_vector = user_vector.reindex(columns=X.columns, fill_value=0)

    probabilities = clf.predict_proba(user_vector)[0]
    best_country_idx = probabilities.argmax()
    best_country = countries[best_country_idx]

    best_country_info = df_countries[df_countries['country'] == best_country].iloc[0]
    return best_country, best_country_info, probabilities

# Configure Gemini AI
GOOGLE_API_KEY = 'AIzaSyC0wSDnog0hlXNY__FYeObQPqPXYbXplcs'  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def gemini_chat(country_info, sport):
    prompt = (f"Provide insights about the country with the best performance in Paris 2024 Olympics in {sport}. "
              f"Here are the details: Country: {country_info['country']}, "
              f"Total Medals: {country_info['total_medals']}, "
              f"Number of Sports: {country_info['number_of_sports']}. "
              f"Discuss strengths, weaknesses, and how well the country performed in Paris 2024 Olympics.")
    response = model.generate_content(prompt)
    return response.text

def comparison_chart(sport, best_country, probabilities, year):
    top_indices = probabilities.argsort()[-5:][::-1]
    top_countries = [countries[i] for i in top_indices]
    top_probabilities = probabilities[top_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(top_countries, top_probabilities, color='skyblue')
    plt.title(f'Comparison of Top Countries for {sport} ({year})')
    plt.xlabel('Country')
    plt.ylabel('Probability')
    plt.grid(True)
    st.pyplot(plt)

def show_summary_card(country_info):
    st.markdown(f"### Country: **{country_info['country']}**")
    st.markdown(f"**Total Medals:** {country_info['total_medals']}")
    st.markdown(f"**Number of Sports Competed:** {country_info['number_of_sports']}")
    st.markdown(f"**Top Sport:** {country_info['best_sport']}")
    st.markdown(f"**Number of Athletes:** {country_info['num_athletes']}")

# Streamlit interface
st.title('Olympic Sports Recommendation and Analysis')

# Adding user-friendly widgets
st.write("Welcome to the Olympic Sports Recommendation System! Let's find out which country excels in your favorite sport.")
st.image("https://th.bing.com/th/id/OIP.TlN8amZfCgC7Ejq7aM2_UQAAAA?rs=1&pid=ImgDetMain")  # Add an engaging banner image (replace with actual image URL)

sport_to_test = st.selectbox('Select a sport to analyze', sports)
year_option = st.selectbox('Select the Olympic year', ['Tokyo 2020', 'Paris 2024'])

if st.button('Get Best Country'):
    country, country_info, probabilities = get_best_country(sport_to_test)

    if country_info is not None:
        # Show Summary Card
        show_summary_card(country_info)
        
        # Generate and display insights using Gemini AI
        response = gemini_chat(country_info, sport_to_test)
        st.markdown("### AI Insights:")
        st.write(response)
        
        # Comparison Chart
        st.markdown(f"### How does {country} compare with other top countries in {sport_to_test} in {year_option}?")
        comparison_chart(sport_to_test, country, probabilities, year_option)
    else:
        st.error("Sport not recognized. Please select a valid sport.")
