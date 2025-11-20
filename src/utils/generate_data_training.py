from datetime import timedelta
import fastf1
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from collections import defaultdict
from datetime import datetime

fastf1.Cache.enable_cache("f1_cache",use_requests_cache=True)
fastf1.Cache.offline_mode(enabled=True) # Comment this for making request to the API

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

HISTORICAL_RACES = [
    # 2018
    (2018, "Australian Grand Prix", -37.8497, 144.9680),
    (2018, "Bahrain Grand Prix", 26.0325, 50.5106),
    (2018, "Chinese Grand Prix", 31.3389, 121.2200),
    (2018, "Azerbaijan Grand Prix", 40.3725, 49.8533),
    (2018, "Spanish Grand Prix", 41.5700, 2.2611),
    (2018, "Monaco Grand Prix", 43.7347, 7.42056),
    (2018, "Canadian Grand Prix", 45.5000, -73.5228),
    (2018, "French Grand Prix", 43.2506, 5.79139),
    (2018, "Austrian Grand Prix", 47.2192, 14.7647),
    (2018, "British Grand Prix", 52.0786, -1.01694),
    (2018, "German Grand Prix", 49.2539, 8.8986),
    (2018, "Hungarian Grand Prix", 47.5789, 19.2486),
    (2018, "Belgian Grand Prix", 50.5106, 5.25694),
    (2018, "Italian Grand Prix", 45.6150, 9.28111),
    (2018, "Singapore Grand Prix", 1.2914, 103.8640),
    (2018, "Russian Grand Prix", 43.4056, 39.9578),
    (2018, "Japanese Grand Prix", 34.8431, 136.5410),
    (2018, "United States Grand Prix", 30.1328, -97.6411),
    (2018, "Mexico City Grand Prix", 19.4042, -99.0906),
    (2018, "São Paulo Grand Prix", -23.7036, -46.6997),
    (2018, "Abu Dhabi Grand Prix", 24.4672, 54.6031),

    # 2019–2022 (same naming style)
    (2019, "Australian Grand Prix", -37.8497, 144.9680),
    (2019, "Bahrain Grand Prix", 26.0325, 50.5106),
    (2019, "Chinese Grand Prix", 31.3389, 121.2200),
    (2019, "Azerbaijan Grand Prix", 40.3725, 49.8533),
    (2019, "Spanish Grand Prix", 41.5700, 2.2611),
    (2019, "Monaco Grand Prix", 43.7347, 7.42056),
    (2019, "Canadian Grand Prix", 45.5000, -73.5228),
    (2019, "French Grand Prix", 43.2506, 5.79139),
    (2019, "Austrian Grand Prix", 47.2192, 14.7647),
    (2019, "British Grand Prix", 52.0786, -1.01694),
    (2019, "German Grand Prix", 49.2539, 8.8986),
    (2019, "Hungarian Grand Prix", 47.5789, 19.2486),
    (2019, "Belgian Grand Prix", 50.5106, 5.25694),
    (2019, "Italian Grand Prix", 45.6150, 9.28111),
    (2019, "Singapore Grand Prix", 1.2914, 103.8640),
    (2019, "Russian Grand Prix", 43.4056, 39.9578),
    (2019, "Japanese Grand Prix", 34.8431, 136.5410),
    (2019, "Mexico City Grand Prix", 19.4042, -99.0906),
    (2019, "United States Grand Prix", 30.1328, -97.6411),
    (2019, "São Paulo Grand Prix", -23.7036, -46.6997),
    (2019, "Abu Dhabi Grand Prix", 24.4672, 54.6031),

    # 2020 (COVID calendar – some special names remain as-is because fastf1 uses them)
    (2020, "Austrian Grand Prix", 47.2192, 14.7647),
    (2020, "Styrian Grand Prix", 47.2192, 14.7647),
    (2020, "Hungarian Grand Prix", 47.5789, 19.2486),
    (2020, "British Grand Prix", 52.0786, -1.01694),
    (2020, "70th Anniversary Grand Prix", 52.0786, -1.01694),
    (2020, "Spanish Grand Prix", 41.5700, 2.2611),
    (2020, "Belgian Grand Prix", 50.5106, 5.25694),
    (2020, "Italian Grand Prix", 45.6150, 9.28111),
    (2020, "Tuscan Grand Prix", 43.9978, 11.3719),
    (2020, "Russian Grand Prix", 43.4056, 39.9578),
    (2020, "Eifel Grand Prix", 50.3347, 6.9475),
    (2020, "Portuguese Grand Prix", 38.7500, -9.3947),
    (2020, "Emilia Romagna Grand Prix", 44.3439, 11.7167),
    (2020, "Turkish Grand Prix", 41.0850, 28.7878),
    (2020, "Bahrain Grand Prix", 26.0325, 50.5106),
    (2020, "Sakhir Grand Prix", 26.0325, 50.5106),
    (2020, "Abu Dhabi Grand Prix", 24.4672, 54.6031),

    # 2021–2024
    (2021, "Bahrain Grand Prix", 26.0325, 50.5106),
    (2021, "Emilia Romagna Grand Prix", 44.3439, 11.7167),
    (2021, "Portuguese Grand Prix", 38.7500, -9.3947),
    (2021, "Spanish Grand Prix", 41.5700, 2.2611),
    (2021, "Monaco Grand Prix", 43.7347, 7.42056),
    (2021, "Azerbaijan Grand Prix", 40.3725, 49.8533),
    (2021, "French Grand Prix", 43.2506, 5.79139),
    (2021, "Styrian Grand Prix", 47.2192, 14.7647),
    (2021, "Austrian Grand Prix", 47.2192, 14.7647),
    (2021, "British Grand Prix", 52.0786, -1.01694),
    (2021, "Hungarian Grand Prix", 47.5789, 19.2486),
    (2021, "Belgian Grand Prix", 50.5106, 5.25694),
    (2021, "Dutch Grand Prix", 52.3888, 4.54092),
    (2021, "Italian Grand Prix", 45.6150, 9.28111),
    (2021, "Russian Grand Prix", 43.4056, 39.9578),
    (2021, "Turkish Grand Prix", 41.0850, 28.7878),
    (2021, "United States Grand Prix", 30.1328, -97.6411),
    (2021, "Mexico City Grand Prix", 19.4042, -99.0906),
    (2021, "São Paulo Grand Prix", -23.7036, -46.6997),
    (2021, "Qatar Grand Prix", 25.4900, 51.4542),
    (2021, "Saudi Arabian Grand Prix", 21.6319, 39.1044),
    (2021, "Abu Dhabi Grand Prix", 24.4672, 54.6031),

    (2022, "Bahrain Grand Prix", 26.0325, 50.5106),
    (2022, "Saudi Arabian Grand Prix", 21.6319, 39.1044),
    (2022, "Australian Grand Prix", -37.8497, 144.9680),
    (2022, "Emilia Romagna Grand Prix", 44.3439, 11.7167),
    (2022, "Miami Grand Prix", 25.9581, -80.2389),
    (2022, "Spanish Grand Prix", 41.5700, 2.2611),
    (2022, "Monaco Grand Prix", 43.7347, 7.42056),
    (2022, "Azerbaijan Grand Prix", 40.3725, 49.8533),
    (2022, "Canadian Grand Prix", 45.5000, -73.5228),
    (2022, "British Grand Prix", 52.0786, -1.01694),
    (2022, "Austrian Grand Prix", 47.2192, 14.7647),
    (2022, "French Grand Prix", 43.2506, 5.79139),
    (2022, "Hungarian Grand Prix", 47.5789, 19.2486),
    (2022, "Belgian Grand Prix", 50.5106, 5.25694),
    (2022, "Dutch Grand Prix", 52.3888, 4.54092),
    (2022, "Italian Grand Prix", 45.6150, 9.28111),
    (2022, "Singapore Grand Prix", 1.2914, 103.8640),
    (2022, "Japanese Grand Prix", 34.8431, 136.5410),
    (2022, "United States Grand Prix", 30.1328, -97.6411),
    (2022, "Mexico City Grand Prix", 19.4042, -99.0906),
    (2022, "São Paulo Grand Prix", -23.7036, -46.6997),
    (2022, "Abu Dhabi Grand Prix", 24.4672, 54.6031),

    (2023, "Bahrain Grand Prix", 26.0325, 50.5106),
    (2023, "Saudi Arabian Grand Prix", 21.6319, 39.1044),
    (2023, "Australian Grand Prix", -37.8497, 144.9680),
    (2023, "Azerbaijan Grand Prix", 40.3725, 49.8533),
    (2023, "Miami Grand Prix", 25.9581, -80.2389),
    (2023, "Monaco Grand Prix", 43.7347, 7.4206),
    (2023, "Spanish Grand Prix", 41.5700, 2.2611),
    (2023, "Canadian Grand Prix", 45.5000, -73.5228),
    (2023, "Austrian Grand Prix", 47.2197, 14.7647),
    (2023, "British Grand Prix", 52.0786, -1.0169),
    (2023, "Hungarian Grand Prix", 47.5789, 19.2486),
    (2023, "Belgian Grand Prix", 50.4372, 5.9714),
    (2023, "Dutch Grand Prix", 52.3888, 4.5409),
    (2023, "Italian Grand Prix", 45.6156, 9.2811),
    (2023, "Singapore Grand Prix", 1.2914, 103.8640),
    (2023, "Japanese Grand Prix", 34.8431, 136.5408),
    (2023, "Qatar Grand Prix", 25.4900, 51.4542),
    (2023, "United States Grand Prix", 30.1328, -97.6411),
    (2023, "Mexico City Grand Prix", 19.4042, -99.0907),
    (2023, "São Paulo Grand Prix", -23.7036, -46.6997),
    (2023, "Las Vegas Grand Prix", 36.1147, -115.1728),
    (2023, "Abu Dhabi Grand Prix", 24.4672, 54.6031),

    (2024, "Bahrain Grand Prix", 26.0325, 50.5106),
    (2024, "Saudi Arabian Grand Prix", 21.6319, 39.1044),
    (2024, "Australian Grand Prix", -37.8497, 144.9680),
    (2024, "Japanese Grand Prix", 34.8431, 136.5408),
    (2024, "Chinese Grand Prix", 31.3389, 121.2200),
    (2024, "Miami Grand Prix", 25.9581, -80.2389),
    (2024, "Emilia Romagna Grand Prix", 44.3439, 11.7167),
    (2024, "Monaco Grand Prix", 43.7347, 7.4206),
    (2024, "Canadian Grand Prix", 45.5000, -73.5228),
    (2024, "Spanish Grand Prix", 41.5700, 2.2611),
    (2024, "Austrian Grand Prix", 47.2197, 14.7647),
    (2024, "British Grand Prix", 52.0786, -1.0169),
    (2024, "Hungarian Grand Prix", 47.5789, 19.2486),
    (2024, "Belgian Grand Prix", 50.4372, 5.9714),
    (2024, "Dutch Grand Prix", 52.3888, 4.5409),
    (2024, "Italian Grand Prix", 45.6156, 9.2811),
    (2024, "Azerbaijan Grand Prix", 40.3725, 49.8533),
    (2024, "Singapore Grand Prix", 1.2914, 103.8640),
    (2024, "United States Grand Prix", 30.1328, -97.6411),
    (2024, "Mexico City Grand Prix", 19.4042, -99.0907),
    (2024, "São Paulo Grand Prix", -23.7036, -46.6997),
    (2024, "Las Vegas Grand Prix", 36.1147, -115.1728),
    (2024, "Qatar Grand Prix", 25.4900, 51.4542),
    (2024, "Abu Dhabi Grand Prix", 24.4672, 54.6031),

    (2025, "Bahrain Grand Prix", 26.0325, 50.5106),
    (2025, "Saudi Arabian Grand Prix", 21.6319, 39.1044),
    (2025, "Australian Grand Prix", -37.8497, 144.9680),
    (2025, "Japanese Grand Prix", 34.8431, 136.5408),
    (2025, "Chinese Grand Prix", 31.3389, 121.2200),
    (2025, "Miami Grand Prix", 25.9581, -80.2389),
    (2025, "Emilia Romagna Grand Prix", 44.3439, 11.7167),
    (2025, "Monaco Grand Prix", 43.7347, 7.4206),
    (2025, "Canadian Grand Prix", 45.5000, -73.5228),
    (2025, "Spanish Grand Prix", 41.5700, 2.2611),
    (2025, "Austrian Grand Prix", 47.2197, 14.7647),
    (2025, "British Grand Prix", 52.0786, -1.0169),
    (2025, "Hungarian Grand Prix", 47.5789, 19.2486),
    (2025, "Belgian Grand Prix", 50.4372, 5.9714),
    (2025, "Dutch Grand Prix", 52.3888, 4.5409),
    (2025, "Italian Grand Prix", 45.6156, 9.2811),
    (2025, "Azerbaijan Grand Prix", 40.3725, 49.8533),
    (2025, "Singapore Grand Prix", 1.2914, 103.8640),
    (2025, "United States Grand Prix", 30.1328, -97.6411),
    (2025, "Mexico City Grand Prix", 19.4042, -99.0907),
    #(2025, "São Paulo Grand Prix", -23.7036, -46.6997),
    #(2025, "Las Vegas Grand Prix", 36.1147, -115.1728),
    #(2025, "Qatar Grand Prix", 25.4900, 51.4542),
    #(2025, "Abu Dhabi Grand Prix", 24.4672, 54.6031),
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
    
    # Get team name
    team_map = quali.results.set_index('Abbreviation')['TeamName'].to_dict()

    # Load race session
    race = fastf1.get_session(year, gp_name, "R")
    race.load()
    
    # Get qualifying results (grid position)
    quali_results = _get_qualifyin_results(quali)
    
    # Add Team column using the driver's abbreviation
    quali_results['Team'] = quali_results['Driver'].map(team_map)

    # Get race avg lap times
    avg_race_times = _get_race_lap_times(race)
    
    # Get weather data from race
    race_data = _get_weather_from_race(quali_results,avg_race_times,race,lat,lon)

    # Add metadata
    date = race.session_info['StartDate']
    race_data["Date"] = date.strftime("%Y-%m-%d")
    race_data["Year"] = year
    race_data["GP"] = gp_name
    race_data["Laps"] = race.total_laps
    
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

def _get_drivers_elo(historical_races_data):

    df = historical_races_data.copy()
    df = df[['Year', 'GP', 'Driver', 'QualifyingTime (s)']].dropna(subset=['QualifyingTime (s)'])
    
    # Sort chronologically
    df['RaceDate'] = df.apply(lambda row: fastf1.get_session(row['Year'], row['GP'], 'R').date, axis=1)
    df = df.sort_values('RaceDate').reset_index(drop=True)
    
    K = 32  # Standard K-factor (you can tune: higher = more volatile)
    INITIAL_ELO = 1500
    
    driver_elo = defaultdict(lambda: INITIAL_ELO)
    elo_history = []
    
    for (year, gp), group in df.groupby(['Year', 'GP'], sort=False):
        # Current Elo before this race
        before_elo = {drv: driver_elo[drv] for drv in group['Driver']}
        
        # Rank drivers by qualifying time (lower time = better)
        group = group.sort_values('QualifyingTime (s)')
        group['rank'] = np.arange(1, len(group) + 1)
        
        drivers = group['Driver'].tolist()
        expected_matrix = {}
        actual_matrix = {}
        
        # Pre-calculate expected win probabilities
        for i, drv_i in enumerate(drivers):
            for j, drv_j in enumerate(drivers):
                if i == j:
                    continue
                rating_diff = before_elo[drv_j] - before_elo[drv_i]
                expected = 1 / (1 + 10 ** (rating_diff / 400))
                expected_matrix[(drv_i, drv_j)] = expected
                # Actual result: 1 if i beat j in quali, 0.5 if tie (rare), 0 otherwise
                actual = 1.0 if group.loc[group['Driver'] == drv_i, 'QualifyingTime (s)'].iloc[0] < \
                                 group.loc[group['Driver'] == drv_j, 'QualifyingTime (s)'].iloc[0] else 0.0
                if abs(group.loc[group['Driver'] == drv_i, 'QualifyingTime (s)'].iloc[0] - 
                       group.loc[group['Driver'] == drv_j, 'QualifyingTime (s)'].iloc[0]) < 0.001:
                    actual = 0.5
                actual_matrix[(drv_i, drv_j)] = actual
        
        # Update Elo for each driver
        new_elo = before_elo.copy()
        for drv in drivers:
            elo_change = 0
            opponents = [d for d in drivers if d != drv]
            for opp in opponents:
                expected = expected_matrix.get((drv, opp), 1 - expected_matrix.get((opp, drv), 0.5))
                actual = actual_matrix.get((drv, opp), 1 - actual_matrix.get((opp, drv), 0.5))
                elo_change += K * (actual - expected)
            new_elo[drv] += elo_change
            driver_elo[drv] = new_elo[drv]
        
        # Record
        for drv in drivers:
            elo_history.append({
                'Year': year,
                'GP': gp,
                'Driver': drv,
                'DriverEloBefore': before_elo[drv],
                'DriverEloAfter': new_elo[drv],
                'QualifierPosition': group.loc[group['Driver'] == drv, 'rank'].iloc[0]
            })
    
    elo_df = pd.DataFrame(elo_history)
    return elo_df[['Year', 'GP', 'Driver', 'DriverEloBefore', 'DriverEloAfter']]


def _get_constructors_elo(historical_races_data):

    df = historical_races_data.copy()
    df = df[['Year', 'GP', 'Driver', 'QualifyingTime (s)']].dropna(subset=['QualifyingTime (s)'])

    # === Add Team information using fastf1 ===
    team_mappings = []
    for (year, gp), group in df.groupby(['Year', 'GP']):
        session = fastf1.get_session(year, gp, 'Q')
        try:
            session.load(telemetry=False, weather=False, messages=False)
            team_map = session.results.set_index('Abbreviation')['TeamName'].to_dict()
        except:
            # Fallback if session fails to load
            team_map = {}
        group = group.copy()
        group['Team'] = group['Driver'].map(team_map)
        team_mappings.append(group)
    df = pd.concat(team_mappings, ignore_index=True)
    df = df.dropna(subset=['Team'])

    # Keep only the fastest driver per team per race
    best_per_team = df.loc[df.groupby(['Year', 'GP', 'Team'])['QualifyingTime (s)'].idxmin()]

    # Sort races chronologically
    best_per_team['RaceDate'] = best_per_team.apply(
        lambda row: fastf1.get_session(row['Year'], row['GP'], 'R').date, axis=1
    )
    best_per_team = best_per_team.sort_values('RaceDate').reset_index(drop=True)

    K = 32
    INITIAL_ELO = 1500
    team_elo = defaultdict(lambda: INITIAL_ELO)
    elo_records = []

    for (year, gp), group in best_per_team.groupby(['Year', 'GP'], sort=False):
        teams = group['Team'].unique()
        before_elo = {team: team_elo[team] for team in teams}

        # Sort by qualifying time (faster = better)
        group = group.sort_values('QualifyingTime (s)').reset_index(drop=True)

        # Update Elo for every team in this race
        elo_change = {team: 0.0 for team in teams}

        for i, team_i in enumerate(group['Team']):
            for j, team_j in enumerate(group['Team']):
                if i == j:
                    continue

                # Expected win probability for team_i over team_j
                expected = 1 / (1 + 10 ** ((before_elo[team_j] - before_elo[team_i]) / 400))

                # Actual result
                time_i = group.iloc[i]['QualifyingTime (s)']
                time_j = group.iloc[j]['QualifyingTime (s)']
                actual = 1.0 if time_i < time_j else 0.0
                if abs(time_i - time_j) < 0.001:  # rare exact tie
                    actual = 0.5

                elo_change[team_i] += K * (actual - expected)

        # Apply changes
        for team in teams:
            new_rating = before_elo[team] + elo_change[team]
            team_elo[team] = new_rating

            elo_records.append({
                'Year': year,
                'GP': gp,
                'Team': team,
                'ConstructorEloBefore': before_elo[team],
                'ConstructorEloAfter': new_rating
            })

    return pd.DataFrame(elo_records)

def get_drivers_constructors_elo(training_data):
    driver_elo_df = _get_drivers_elo(training_data) # Get drivers elo
    constructor_elo_df = _get_constructors_elo(training_data) # Get constructor elo

    training_data = training_data.merge(
        driver_elo_df[['Year', 'GP', 'Driver', 'DriverEloBefore']],
        on=['Year', 'GP', 'Driver'],
        how='left'
    )

    training_data = training_data.merge(
        constructor_elo_df[['Year', 'GP', 'Team', 'ConstructorEloBefore']],
        on=['Year', 'GP', 'Team'],
        how='left'
    )

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

    # Get drivers and drconstructors elo
    training_data = get_drivers_constructors_elo(training_data)

    # Verify NaN values
    training_data = verify_nan_and_fill_with_mean_values(training_data)
    
    # Save dataset
    save_dataset(training_data)