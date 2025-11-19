from datetime import timedelta
import fastf1
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

fastf1.Cache.enable_cache("f1_cache")

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

HISTORICAL_RACES = [
    (2023, "Bahrain", 26.0325, 50.5106),
    (2023, "Saudi Arabia", 21.6319, 39.1044),
    (2023, "Australia", -37.8497, 144.9680),
    (2023, "China", 31.3389, 121.2200),
    (2023, "Azerbaijan", 40.3725, 49.8533),
    (2023, "Miami", 25.9581, -80.2389),
    (2023, "Emilia‑Romagna", 44.3439, 11.7167),
    (2023, "Monaco", 43.7347, 7.4206),
    (2023, "Spain", 41.5700, 2.2611),
    (2023, "Canada", 45.5000, -73.5228),
    (2023, "Austria", 47.2197, 14.7647),
    (2023, "United Kingdom", 52.0786, -1.0169),
    (2023, "Hungary", 47.5789, 19.2486),
    (2023, "Belgium", 50.4372, 5.9714),
    (2023, "Netherlands", 52.3888, 4.5409),
    (2023, "Italy", 45.6156, 9.2811),
    (2023, "Singapore", 1.2914, 103.8640),
    (2023, "Japan", 34.8431, 136.5408),
    (2023, "Qatar", 25.4900, 51.4542),
    (2023, "United States", 30.1328, -97.6411),
    (2023, "Mexico", 19.4042, -99.0907),
    (2023, "Brazil", -23.7036, -46.6997),
    (2023, "Las Vegas", 36.1147, -115.1728),
    (2023, "Abu Dhabi", 24.4672, 54.6031),
    (2024, "Bahrain", 26.0325, 50.5106),
    (2024, "Saudi Arabia", 21.6319, 39.1044),
    (2024, "Australia", -37.8497, 144.9680),
    (2024, "Japan", 34.8431, 136.5408),
    (2024, "China", 31.3389, 121.2200),
    (2024, "Miami", 25.9581, -80.2389),
    (2024, "Emilia‑Romagna", 44.3439, 11.7167),
    (2024, "Monaco", 43.7347, 7.4206),
    (2024, "Canada", 45.5000, -73.5228),
    (2024, "Spain", 41.5700, 2.2611),
    (2024, "Austria", 47.2197, 14.7647),
    (2024, "United Kingdom", 52.0786, -1.0169),
    (2024, "Hungary", 47.5789, 19.2486),
    (2024, "Belgium", 50.4372, 5.9714),
    (2024, "Netherlands", 52.3888, 4.5409),
    (2024, "Italy", 45.6156, 9.2811),
    (2024, "Azerbaijan", 40.3725, 49.8533),
    (2024, "Singapore", 1.2914, 103.8640),
    (2024, "United States", 30.1328, -97.6411),
    (2024, "Mexico", 19.4042, -99.0907),
    (2024, "Brazil", -23.7036, -46.6997),
    (2024, "Las Vegas", 36.1147, -115.1728),
    (2024, "Qatar", 25.4900, 51.4542),
    (2024, "Abu Dhabi", 24.4672, 54.6031),
    (2025, "Sakhir", 26.0325, 50.5106),
    (2025, "Australia", -37.8497, 144.9680),
    (2025, "China", 31.3389, 121.2200),
    (2025, "Japan", 34.8431, 136.5408),
    (2025, "Bahrain", 26.0325, 50.5106),
    (2025, "Saudi Arabia", 21.6319, 39.1044),
    (2025, "Miami", 25.9581, -80.2389),
    (2025, "Emilia‑Romagna", 44.3439, 11.7167),
    (2025, "Monaco", 43.7347, 7.4206),
    (2025, "Spain", 41.5700, 2.2611),
    (2025, "Canada", 45.5000, -73.5228),
    (2025, "Austria", 47.2197, 14.7647),
    (2025, "United Kingdom", 52.0786, -1.0169),
    (2025, "Belgium", 50.4372, 5.9714),
    (2025, "Hungary", 47.5789, 19.2486),
    (2025, "Netherlands", 52.3888, 4.5409),
    (2025, "Italy", 45.6156, 9.2811),
    (2025, "Azerbaijan", 40.3725, 49.8533),
    (2025, "Singapore", 1.2914, 103.8640),
    (2025, "United States", 30.1328, -97.6411),
    (2025, "Mexico", 19.4042, -99.0907),
    #(2025, "Brazil", -23.7036, -46.6997),
]

def _get_qualifyin_results(quali):
    quali_results = quali.results[["Abbreviation", "Q3", "Q2", "Q1"]].copy()
    quali_results["QualifyingTime"] = quali_results["Q3"].fillna(quali_results["Q2"]).fillna(quali_results["Q1"])
    quali_results["QualifyingTime (s)"] = quali_results["QualifyingTime"].dt.total_seconds()
    quali_results = quali_results[["Abbreviation", "QualifyingTime (s)"]].rename(columns={"Abbreviation": "Driver"})
    return quali_results

def _get_race_lap_times(race): 
    # Get race lap times
    race_laps = race.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    race_laps.dropna(inplace=True)
    
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        race_laps[f"{col} (s)"] = race_laps[col].dt.total_seconds()
    
    # Average race lap time per driver (target variable)
    avg_race_times = race_laps.groupby("Driver")["LapTime (s)"].mean().reset_index()
    avg_race_times.columns = ["Driver", "AvgRaceTime (s)"]
    return avg_race_times

def _get_average_rain_from_open_meteo(race,lat,lon):

    session_info = race.session_info
    start_date = session_info['StartDate']
    end_date = session_info['EndDate']

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "rain",
        "timezone": "auto",
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

    start_date_utc = start_date.astimezone(pd.Timestamp('now', tz='UTC').tzinfo)
    end_date_utc = end_date.astimezone(pd.Timestamp('now', tz='UTC').tzinfo)

    # Add error to ours because we have weather metrics by hour. If its :30 we dont get full data
    start_date_utc = start_date_utc - timedelta(hours=1) 
    end_date_utc = end_date_utc + timedelta(hours=1)

    race_period = hourly_dataframe[
        (hourly_dataframe['date'] >= start_date_utc) &
        (hourly_dataframe['date'] <= end_date_utc)
    ]

    return race_period['rain'].mean()

def _get_weather_from_race(quali_results,avg_race_times,race,lat,lon):
    weather = race.weather_data
    if weather is not None and len(weather) > 0:
        avg_temp = weather["AirTemp"].mean()
        avg_humidity = weather["Humidity"].mean()
        avg_pressure = weather["Pressure"].mean()
        rainfall = _get_average_rain_from_open_meteo(race,lat,lon)
        avg_track_temp = weather["TrackTemp"].mean()
    else:
        avg_temp, avg_humidity, avg_pressure, rainfall, avg_track_temp = -1, -1, -1, -1, -1 # No date has these values. However always verify.
    
    # Merge qualifying and race data
    race_data = quali_results.merge(avg_race_times, on="Driver", how="inner")
    
    # Add weather features (same for all drivers in this race)
    race_data["Temperature (°C)"] = avg_temp
    race_data["Humidity (%)"] = avg_humidity
    race_data["Pressure (mbar)"] = avg_pressure
    race_data["Rainfall (mm)"] = rainfall
    race_data["TrackTemp (°C)"] = avg_track_temp

    return race_data

def _extract_race_features(year, gp_name, lat, lon):
    
    # Load qualifying session
    quali = fastf1.get_session(year, gp_name, "Q")
    quali.load()
    
    # Load race session
    race = fastf1.get_session(year, gp_name, "R")
    race.load()
    
    # Get qualifying results (grid position)
    quali_results = _get_qualifyin_results(quali)
    
    # Get race avg lap times
    avg_race_times = _get_race_lap_times(race)
    
    # Get weather data from race
    race_data = _get_weather_from_race(quali_results,avg_race_times,race,lat,lon)

    # Add metadata
    race_data["Year"] = year
    race_data["GP"] = gp_name
    
    return race_data

def get_historical_races_data():
    races_data = []
    for year, gp, lat, lon in HISTORICAL_RACES:
        try:
            race_features = _extract_race_features(year, gp, lat, lon)
            races_data.append(race_features)
        except Exception as e:
            print(f"⚠️ Skipped {year} {gp}: {e}")
    return races_data

def get_drivers_average_performance(training_data):
    driver_avg_times = training_data.groupby("Driver")["AvgRaceTime (s)"].mean().to_dict()
    return training_data["Driver"].map(driver_avg_times)

def add_driver_track_condition_performance(training_data):
    target = "AvgRaceTime (s)"

    conditions = {
        "Dry_Perf": training_data["Rainfall (mm)"] < 0.1,
        "Damp_Perf": (training_data["Rainfall (mm)"] >= 0.1) & (training_data["Rainfall (mm)"] <= 0.5),
        "Wet_Perf": training_data["Rainfall (mm)"] > 0.5,
        "Cold_Perf": training_data["TrackTemp (°C)"] < 22,
        "Mid_Perf": training_data["TrackTemp (°C)"] >= 22 & (training_data["TrackTemp (°C)"] <= 45),
        "Hot_Perf": training_data["TrackTemp (°C)"] > 45,
        "HighDownforce_Perf": training_data["GP"].isin(["Monaco", "Singapore", "Hungary", "Azerbaijan", "Mexico"]),
        "LowDownforce_Perf": training_data["GP"].isin(["Monza", "Spa", "Silverstone", "Saudi Arabia", "Canada"]),
    }

    for name, mask in conditions.items():
        subset = training_data[mask]
        if len(subset) == 0:
            training_data[f"Driver_{name}"] = 50.0  # neutral
            continue

        # Percentile rank PER RACE (100 = fastest in that race)
        subset["pct"] = 100 - (subset.groupby(["Year", "GP"])[target].rank(method='min', ascending=True, pct=True) * 100)

        driver_perf = subset.groupby("Driver")["pct"].mean()
        training_data[f"Driver_{name}"] = training_data["Driver"].map(driver_perf).fillna(50.0)

    return training_data

def verify_nan_and_fill_with_mean_values(training_data):
    if(training_data.isnull().any().any()):
        return _fill_nan_values_with_mean(training_data)
    return training_data

def _fill_nan_values_with_mean(training_data):
    for col in training_data.columns:
        if training_data[col].isnull().any():
            median_val = training_data[col].median()
            training_data[col].fillna(median_val, inplace=True)
    return training_data

def save_dataset(training_data):
    df = pd.DataFrame(training_data)
    df.to_csv("src/generated_data/races_data.csv", index=False)

if __name__=="__main__":
    races_data = get_historical_races_data() # Get historical races history
    
    training_data = pd.concat(races_data, ignore_index=True)

    # Calculate each driver's average performance across all races
    training_data["DriverHistoricalAvg (s)"] = get_drivers_average_performance(training_data) # Get drivers average performance
    training_data = add_driver_track_condition_performance(training_data) # Get drivers average performance in specific weather

    print(training_data)

    # Verify NaN values
    training_data = verify_nan_and_fill_with_mean_values(training_data)
    
    # Save dataset
    save_dataset(training_data)