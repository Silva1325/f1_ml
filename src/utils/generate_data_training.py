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
fastf1.Cache.offline_mode(enabled=True)

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# F1 Points system (current, used since 2010)
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

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
]

def _get_qualifyin_data(year, gp_name):
    quali = fastf1.get_session(year, gp_name, "Q")
    quali.load()

    quali_results = quali.results[["Abbreviation", "Q3", "Q2", "Q1"]].copy()

    quali_results["QualifyingTime"] = (quali_results["Q3"].fillna(quali_results["Q2"]).fillna(quali_results["Q1"]))

    quali_results["QualifyingTime (s)"] = quali_results["QualifyingTime"].dt.total_seconds()

    quali_results["QualifyingTime (s)"].fillna(
        quali_results["QualifyingTime (s)"].mean(),
        inplace=True
    )

    return quali_results[["Abbreviation", "QualifyingTime (s)"]].rename(columns={"Abbreviation": "Driver"})

def _get_race_general_results(race):
    race_results = race.results[['Abbreviation','TeamName','GridPosition','Position']]
    race_results.rename(columns={"Abbreviation": "Driver", "TeamName":"Team", "GridPosition":"StartPosition", "Position":"EndPosition"}, inplace=True)
    return race_results

def _get_race_average_lap_time(race):
    race_laps = race.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()

    race_laps.dropna(inplace=True)

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        race_laps[f"{col} (s)"] = race_laps[col].dt.total_seconds()

    avg_race_times = (
        race_laps.groupby("Driver")["LapTime (s)"]
        .mean()
        .reset_index()
    )

    avg_race_times.columns = ["Driver", "AvgRaceTime (s)"]
    return avg_race_times

def _get_race_data(year, gp_name):
    race = fastf1.get_session(year, gp_name, "R")
    race.load()

    race_results = _get_race_general_results(race)
    avg_race_times = _get_race_average_lap_time(race)

    race_data = race_results.merge(avg_race_times,on="Driver",how="left")

    return race_data

def _calculate_recent_form(driver_history, team_history, driver, team, window=3):
    driver_recent = driver_history.get(driver, [])
    team_recent = team_history.get(team, [])
    
    # Average of last N races (or fewer if not enough history)
    driver_form = np.mean(driver_recent[-window:]) if driver_recent else np.nan
    team_form = np.mean(team_recent[-window:]) if team_recent else np.nan
    
    return driver_form, team_form

def _extract_gp_features(year, gp_name, lat, lon, driver_points_before, team_points_before, 
                         driver_history, team_history):
    qualifying_data = _get_qualifyin_data(year, gp_name)
    race_data = _get_race_data(year, gp_name)

    merged = race_data.merge(qualifying_data, on="Driver", how="left")

    merged["GP"] = gp_name
    merged["Year"] = year
    
    # Add points before race
    merged["DriverPointsBefore"] = merged["Driver"].map(driver_points_before).fillna(0)
    merged["TeamPointsBefore"] = merged["Team"].map(team_points_before).fillna(0)
    
    # Add recent form features
    merged["DriverRecentForm"] = merged.apply(
        lambda row: _calculate_recent_form(driver_history, team_history, row["Driver"], row["Team"])[0], 
        axis=1
    )
    merged["TeamRecentForm"] = merged.apply(
        lambda row: _calculate_recent_form(driver_history, team_history, row["Driver"], row["Team"])[1], 
        axis=1
    )

    merged = merged.reindex(columns=[
        "GP","Year","Driver","Team",
        "DriverPointsBefore", "TeamPointsBefore",
        "DriverRecentForm", "TeamRecentForm",
        "QualifyingTime (s)","AvgRaceTime (s)",
        "StartPosition","EndPosition"
    ])

    return merged

def _update_points(race_data, driver_points, team_points):
    for _, row in race_data.iterrows():
        driver = row["Driver"]
        team = row["Team"]
        position = row["EndPosition"]
        
        # Add points if driver finished in top 10
        if pd.notna(position) and position <= 10:
            points = POINTS_SYSTEM.get(int(position), 0)
            driver_points[driver] = driver_points.get(driver, 0) + points
            team_points[team] = team_points.get(team, 0) + points
    
    return driver_points, team_points

def _update_history(race_data, driver_history, team_history):
    for _, row in race_data.iterrows():
        driver = row["Driver"]
        team = row["Team"]
        position = row["EndPosition"]
        
        # Only add valid positions
        if pd.notna(position):
            if driver not in driver_history:
                driver_history[driver] = []
            if team not in team_history:
                team_history[team] = []
                
            driver_history[driver].append(position)
            team_history[team].append(position)
    
    return driver_history, team_history

def get_historical_gp_data():
    gp_data = []
    
    current_year = None
    driver_points = {}
    team_points = {}
    driver_history = {}
    team_history = {}
    
    for year, gp, lat, lon in HISTORICAL_RACES:
        # Reset points and history when new season starts
        if current_year != year:
            print(f"\n🏁 Starting season {year}")
            current_year = year
            driver_points = {}
            team_points = {}
            driver_history = {}
            team_history = {}
        
        try:
            # Extract GP features with current points and history
            gp_features = _extract_gp_features(year, gp, lat, lon, driver_points, team_points,
                                               driver_history, team_history)
            gp_data.append(gp_features)
            
            # Update points and history after race
            driver_points, team_points = _update_points(gp_features, driver_points, team_points)
            driver_history, team_history = _update_history(gp_features, driver_history, team_history)
            
            print(f"✅ Processed {year} {gp}")
            
        except Exception as e:
            print(f"⚠️ Skipped {year} {gp}: {e}")
    
    return gp_data

def save_dataset(training_data):
    df = pd.DataFrame(training_data)
    df.to_csv("src/generated_data/training_data.csv", index=False)
    print(f"\n💾 Saved dataset with {len(df)} rows")

if __name__=="__main__":
    gp_data = get_historical_gp_data()
    
    training_data = pd.concat(gp_data, ignore_index=True)
    
    # Save dataset
    save_dataset(training_data)