from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.model_selection import train_test_split
import requests

# Initialize Flask app
app = Flask(__name__)

# Load datasets
hotel_data = pd.read_csv('C:\\Users\\sonu\\Hotel_names.csv', encoding='ISO-8859-1')  # Hotel bookings data
tourist_data = pd.read_csv('C:\\Users\\sonu\\tourist_arrival_data.csv')  # Tourist arrivals data

# API Keys
WEATHER_API_KEY = 'eb1e4f5da56519807d4170b600a6be45'
GEOCODE_API_KEY = 'eb7bf890a59745629415cb86699d74d7'

# Preprocess data function
def preprocess_data(hotel_data, tourist_data):
    hotel_data['Price'] = hotel_data['Price'].replace({',': ''}, regex=True).astype(float)
    tourist_data['Total'] = tourist_data['Total'].replace({'%': ''}, regex=True).astype(float)
    tourist_data['Total'].replace([np.inf, -np.inf], np.nan, inplace=True)
    tourist_data['Total'].fillna(0, inplace=True)
    tourist_data['Total'] = tourist_data['Total'].astype(int)

    # Merge datasets on 'Location' and 'District' fields
    merged_data = pd.merge(hotel_data, tourist_data, left_on='Location', right_on='District', how='inner')

    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_data[['Price', 'Total']])
    return scaled_data

# Process the data
data = preprocess_data(hotel_data, tourist_data)
X = data[:, :-1]
y = data[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Create dictionaries for hotels and places by location
places_by_location = hotel_data.groupby('Location')['NearByPlaces'].apply(lambda x: ', '.join(x)).to_dict()
hotels_by_location = hotel_data.groupby('Location').apply(
    lambda x: list(zip(x['Hotel_Name'], x['Rating'], x['Hotel Link']))
).to_dict()

def get_hotels(location):
    """Return a list of hotels for the given location."""
    hotels = hotels_by_location.get(location, [])
    return hotels

def get_places(location):
    """Return a list of nearby places for the given location."""
    places = places_by_location.get(location, "No places available")
    return places.split(', ') if places != "No places available" else []

# Reshape data for LSTM model
input_shape = (X_train.shape[1], 1)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build and train the LSTM model
lstm_model = build_lstm_model(input_shape)
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
lstm_loss = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Model Loss: {lstm_loss}")

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    from_location = data.get('from_location')
    to_location = data.get('to_location')
    price = float(data.get('price'))

    # Make prediction using the LSTM model
    features = np.array([price]).reshape(1, -1, 1)
    prediction = lstm_model.predict(features)

    # Get weather for the 'To' location
    weather_info = get_weather(to_location)

    # Get distance and travel times between 'From' and 'To' locations
    distance_info = get_distance(from_location, to_location)

    # Get the list of hotels and places for the given location
    hotels = get_hotels(to_location)
    places = get_places(to_location)

    return render_template('index.html',
                           prediction=prediction[0][0],
                           from_location=from_location,
                           to_location=to_location,
                           hotels=hotels,
                           places=places,
                           weather_info=weather_info,
                           distance_info=distance_info)

def get_weather(location):
    """Fetch weather information for the given location."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed']
        }
    else:
        return None
def get_distance(from_location, to_location):
    """Fetch distance between two locations and calculate travel times for different modes of transport."""
    url = f"https://api.opencagedata.com/geocode/v1/json?q={to_location}&key={GEOCODE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            to_lat = data['results'][0]['geometry']['lat']
            to_lng = data['results'][0]['geometry']['lng']

            url = f"https://api.opencagedata.com/geocode/v1/json?q={from_location}&key={GEOCODE_API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    from_lat = data['results'][0]['geometry']['lat']
                    from_lng = data['results'][0]['geometry']['lng']

                    # Haversine formula to calculate distance
                    from math import radians, sin, cos, sqrt, atan2
                    R = 6371  # Radius of the Earth in km
                    lat_diff = radians(to_lat - from_lat)
                    lng_diff = radians(to_lng - from_lng)
                    a = sin(lat_diff / 2) ** 2 + cos(radians(from_lat)) * cos(radians(to_lat)) * sin(lng_diff / 2) ** 2
                    c = 2 * atan2(sqrt(a), sqrt(1 - a))
                    distance = R * c  # Distance in kilometers

                    # Calculate travel times based on the distance
                    bus_speed = 40  # km/h
                    train_speed = 80  # km/h
                    car_speed = 60  # km/h

                    travel_time_bus = distance / bus_speed
                    travel_time_train = distance / train_speed
                    travel_time_car = distance / car_speed

                    return {
                        'distance': round(distance, 2),
                        'travel_time_bus': round(travel_time_bus, 2),
                        'travel_time_train': round(travel_time_train, 2),
                       'travel_time_car': round(travel_time_car, 2)
                    }
    return None

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
