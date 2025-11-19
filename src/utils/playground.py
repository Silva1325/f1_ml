
from datetime import timedelta
import fastf1
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

fastf1.Cache.enable_cache("f1_cache")

race = fastf1.get_session(2021, "Belgium", "R")
race.load()

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)


session_info = race.session_info
start_date = session_info['StartDate']
end_date = session_info['EndDate']

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 50.4372, 
	"longitude": 5.9714,
	"start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
	"hourly": "rain",
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_rain = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["rain"] = hourly_rain

hourly_dataframe = pd.DataFrame(data = hourly_data)

print("\nHourly data\n", hourly_dataframe)

start_date_utc = start_date.astimezone(pd.Timestamp('now', tz='UTC').tzinfo)
end_date_utc = end_date.astimezone(pd.Timestamp('now', tz='UTC').tzinfo)
# Expand the time window by 1 hour on each side
start_date_utc = start_date_utc - timedelta(hours=1)
end_date_utc = end_date_utc + timedelta(hours=1)

race_period = hourly_dataframe[
    (hourly_dataframe['date'] >= start_date_utc) & 
    (hourly_dataframe['date'] <= end_date_utc)
]

race_period['rain'].mean()
print("\nHourly data\n", race_period)
print("\nHourly data\n", race_period['rain'].mean())



with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(race.session_info['StartDate'])
    print(race.session_info['EndDate'])
