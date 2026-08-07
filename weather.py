import requests

LAT = 46.1162
LON = -81.2976
HOUR = 0

URL = "https://api.open-meteo.com/v1/forecast"
PARAMS = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation"
    ],
    "forecast_days": 1
}


def weather_variables(params=PARAMS, url=URL):
    response = requests.get(url, params=params)
    data = response.json()
    hourly = data["hourly"]

    temperature = hourly["temperature_2m"][HOUR]
    humidity = hourly["relative_humidity_2m"][HOUR]
    wind_speed = hourly["wind_speed_10m"][HOUR]
    wind_direction = hourly["wind_direction_10m"][HOUR]
    rain = hourly["precipitation"][HOUR]

    return temperature, humidity, wind_speed, wind_direction, rain
