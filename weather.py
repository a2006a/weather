from flask import Flask, request, jsonify
import requests
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import pytz
from opencage.geocoder import OpenCageGeocode
import geocoder

app = Flask(__name__)

# Load API keys securely
OpenWeather_API_Key = 'f936f8e8a5231a6b7e6da388e54cfc1e'
OpenCage_API_Key = 'd64b7b7f7dbd48e48904d0c7743e42ae'
Base_URL = 'https://api.openweathermap.org/data/2.5/'

# Get accurate location
def get_current_location():
    g = geocoder.ip('me')
    lat, lng = g.latlng
    geocoder_oc = OpenCageGeocode(OpenCage_API_Key)
    result = geocoder_oc.reverse_geocode(lat, lng)
    if result:
        components = result[0]['components']
        city = components.get('city', components.get('town', components.get('village', 'N/A')))
        country = components.get('country', 'N/A')
        with open("debug.log", "a") as f:  
            f.write(f"Detected Country: {country}\n")  

        return city
    else:
        print("Unable to detect location.")
        return None

# Get current weather
def get_current_weather(city):
    url = f"{Base_URL}weather?q={city}&appid={OpenWeather_API_Key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        return {
            'city': data['name'],
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'description': data['weather'][0]['description'].capitalize(),
            'pressure': data['main']['pressure'],
            'WindGustSpeed': data['wind']['speed'],
            'wind_gust_dir': data['wind']['deg'],
        }
    except requests.exceptions.HTTPError as e:
        app.logger.error(f"HTTP Error: {e}")
        return None

# Train models once on startup
def load_and_train_models():
    csv_path = 'weather.csv'
    df = pd.read_csv(csv_path).dropna().drop_duplicates()

    # Encode categorical features
    le = LabelEncoder()
    df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
    df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])

    # Train Rain Model
    x_rain = df[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y_rain = df[['RainTomorrow']]
    x_train, x_test, y_train, y_test = train_test_split(x_rain, y_rain, test_size=0.2, random_state=42)
    rain_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rain_model.fit(x_train, y_train.values.ravel())

    # Train Regression Models
    def train_regression(feature):
        x = np.array(df[feature][:-1]).reshape(-1, 1)
        y = np.array(df[feature][1:])
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x, y)
        return model

    return {
        'rain': rain_model,
        'temp': train_regression('Temp'),
        'humidity': train_regression('Humidity'),
        'label_encoder': le
    }

# Initialize models
models = load_and_train_models()

@app.route('/weather', methods=['GET'])
def weather_view():
    city = request.args.get('city') or get_current_location()
    if not city:
        return jsonify({"error": "City name is required"}), 400

    current_weather = get_current_weather(city)
    if current_weather is None:
        return jsonify({"error": "Weather data not available"}), 404

    # Predict Rain
    le = models['label_encoder']
    compass_direction = le.classes_[np.argmin(np.abs(le.transform(le.classes_) - current_weather['wind_gust_dir']))]

    current_data = pd.DataFrame([{
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'WindGustDir': le.transform([compass_direction])[0],
        'WindGustSpeed': current_weather['WindGustSpeed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temp'],
    }])
    
    rain_prediction = models['rain'].predict(current_data)[0]
    rain_probability = round(min(100, max(0, rain_prediction * 100)))

    # Predict future values
    def predict_future(model, current_value):
        predictions = [current_value]
        for _ in range(5):
            next_value = model.predict(np.array([[predictions[-1]]]))[0]
            predictions.append(next_value)
        return [round(val, 1) for val in predictions[1:]]

    future_temp = predict_future(models['temp'], current_weather['temp_min'])
    future_humidity = predict_future(models['humidity'], current_weather['humidity'])

    # Future time formatting
    timezone = pytz.timezone('Asia/Colombo')
    now = datetime.now(timezone)
    future_times = [(now + timedelta(hours=i+1)).strftime("%H:00") for i in range(5)]

    # Response JSON
    response = {
        "city": current_weather['city'],
        "country": "Sri Lanka",
        "current_temp": current_weather['current_temp'],
        "feels_like": current_weather['feels_like'],
        "temp_min": current_weather['temp_min'],
        "temp_max": current_weather['temp_max'],
        "humidity": current_weather['humidity'],
        "description": current_weather['description'],
        "WindGustSpeed": current_weather['WindGustSpeed'],
        "wind_direction": compass_direction,
        "pressure": current_weather['pressure'],
        "rain_probability": rain_probability,
        "will_rain": "Yes" if rain_probability > 50 else "No",
        "future_forecast": {
            "times": future_times,
            "temperature": future_temp,
            "humidity": future_humidity
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
