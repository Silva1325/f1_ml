import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("f1_cache")

def extract_race_features(year, gp_name):
    """Extract features from a historical race"""
    
    # Load qualifying session
    quali = fastf1.get_session(year, gp_name, "Q")
    quali.load()
    
    # Load race session
    race = fastf1.get_session(year, gp_name, "R")
    race.load()
    
    # Get qualifying results (grid position)
    quali_results = quali.results[["Abbreviation", "Q3", "Q2", "Q1"]].copy()
    quali_results["QualifyingTime"] = quali_results["Q3"].fillna(
        quali_results["Q2"]
    ).fillna(quali_results["Q1"])
    quali_results["QualifyingTime (s)"] = quali_results["QualifyingTime"].dt.total_seconds()
    quali_results = quali_results[["Abbreviation", "QualifyingTime (s)"]].rename(
        columns={"Abbreviation": "Driver"}
    )
    
    # Get race lap times
    race_laps = race.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    race_laps.dropna(inplace=True)
    
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        race_laps[f"{col} (s)"] = race_laps[col].dt.total_seconds()
    
    # Average race lap time per driver (target variable)
    avg_race_times = race_laps.groupby("Driver")["LapTime (s)"].mean().reset_index()
    avg_race_times.columns = ["Driver", "AvgRaceTime (s)"]
    
    # Get weather data from race
    weather = race.weather_data
    if weather is not None and len(weather) > 0:
        avg_temp = weather["AirTemp"].mean()
        avg_humidity = weather["Humidity"].mean()
        avg_pressure = weather["Pressure"].mean()
        rainfall = weather["Rainfall"].sum()  # Total rainfall during race
    else:
        avg_temp, avg_humidity, avg_pressure, rainfall = 25, 50, 1013, 0
    
    # Merge qualifying and race data
    race_data = quali_results.merge(avg_race_times, on="Driver", how="inner")
    
    # Add weather features (same for all drivers in this race)
    race_data["Temperature (°C)"] = avg_temp
    race_data["Humidity (%)"] = avg_humidity
    race_data["Pressure (mbar)"] = avg_pressure
    race_data["Rainfall (mm)"] = rainfall
    
    # Add metadata
    race_data["Year"] = year
    race_data["GP"] = gp_name
    
    return race_data

# ============================================
# STEP 1: Build training dataset from multiple historical races
# ============================================

historical_races = [
    (2023, "Bahrain"),
    (2023, "Saudi Arabia"),
    (2023, "Australia"),
    (2023, "China"),
    (2023, "Azerbaijan"),
    (2023, "Miami"),
    (2023, "Emilia‑Romagna"),
    (2023, "Monaco"),
    (2023, "Spain"),
    (2023, "Canada"),
    (2023, "Austria"),
    (2023, "United Kingdom"),
    (2023, "Hungary"),
    (2023, "Belgium"),
    (2023, "Netherlands"),
    (2023, "Italy"),
    (2023, "Singapore"),
    (2023, "Japan"),
    (2023, "Qatar"),
    (2023, "United States"),
    (2023, "Mexico"),
    (2023, "Brazil"),
    (2023, "Las Vegas"),
    (2023, "Abu Dhabi"),
    (2024, "Bahrain"),
    (2024, "Saudi Arabia"),
    (2024, "Australia"),
    (2024, "Japan"),
    (2024, "China"),
    (2024, "Miami"),
    (2024, "Emilia‑Romagna"),
    (2024, "Monaco"),
    (2024, "Canada"),
    (2024, "Spain"),
    (2024, "Austria"),
    (2024, "United Kingdom"),
    (2024, "Hungary"),
    (2024, "Belgium"),
    (2024, "Netherlands"),
    (2024, "Italy"),
    (2024, "Azerbaijan"),
    (2024, "Singapore"),
    (2024, "United States"),
    (2024, "Mexico"),
    (2024, "Brazil"),
    (2024, "Las Vegas"),
    (2024, "Qatar"),
    (2024, "Abu Dhabi"),
    (2025, "Sakhir"),
    (2025, "Australia"),
    (2025, "China"),
    (2025, "Japan"),
    (2025, "Bahrain"),
    (2025, "Saudi Arabia"),
    (2025, "Miami"),
    (2025, "Emilia‑Romagna"),
    (2025, "Monaco"),
    (2025, "Spain"),
    (2025, "Canada"),
    (2025, "Austria"),
    (2025, "United Kingdom"),
    (2025, "Belgium"),
    (2025, "Hungary"),
    (2025, "Netherlands"),
    (2025, "Italy"),
    (2025, "Azerbaijan"),
    (2025, "Singapore"),
    (2025, "United States"),
    (2025, "Mexico"),
    #(2025, "Brazil"), # 22
]


all_race_data = []
for year, gp in historical_races:
    try:
        print(f"Loading {year} {gp}...")
        race_features = extract_race_features(year, gp)
        all_race_data.append(race_features)
    except Exception as e:
        print(f"  ⚠️  Skipped {year} {gp}: {e}")

# Combine all historical data
training_data = pd.concat(all_race_data, ignore_index=True)
print(f"\n✅ Loaded {len(training_data)} driver-race records from {len(all_race_data)} races")

# ============================================
# STEP 2: Add driver performance features
# ============================================

# Calculate each driver's average performance across all races
driver_avg_times = training_data.groupby("Driver")["AvgRaceTime (s)"].mean().to_dict()
training_data["DriverHistoricalAvg (s)"] = training_data["Driver"].map(driver_avg_times)

# Calculate wet weather performance (races with rainfall > 0)
wet_races = training_data[training_data["Rainfall (mm)"] > 0]
if len(wet_races) > 0:
    driver_wet_avg = wet_races.groupby("Driver")["AvgRaceTime (s)"].mean().to_dict()
    training_data["DriverWetAvg (s)"] = training_data["Driver"].map(driver_wet_avg)
else:
    driver_wet_avg = {}  # Empty dict if no wet races
    training_data["DriverWetAvg (s)"] = training_data["DriverHistoricalAvg (s)"]

# Fill missing wet performance with overall average
training_data["DriverWetAvg (s)"].fillna(training_data["DriverHistoricalAvg (s)"], inplace=True)

# Check for any remaining NaN values and fill them
print(f"\n📊 Checking for missing values:")
print(training_data.isnull().sum())

# Fill any remaining NaN with median values
for col in training_data.columns:
    if training_data[col].isnull().any():
        median_val = training_data[col].median()
        training_data[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} NaN values with median: {median_val:.2f}")

# ============================================
# STEP 3: Train the model
# ============================================

X = training_data[[
    "QualifyingTime (s)",
    "DriverHistoricalAvg (s)",
    "DriverWetAvg (s)",
    "Temperature (°C)",
    "Humidity (%)",
    "Pressure (mbar)",
    "Rainfall (mm)"
]]

y = training_data["AvgRaceTime (s)"]

# Double-check for NaN values before training
print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"Missing values in X: {X.isnull().sum().sum()}")
print(f"Missing values in y: {y.isnull().sum()}")

# Remove any rows with NaN in either X or y
valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
X = X[valid_indices]
y = y[valid_indices]

print(f"After removing NaN: {X.shape[0]} samples remaining")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n🔍 Model Error (MAE): {mae:.2f} seconds")

# ============================================
# STEP 4: Predict 2025 Brasil GP
# ============================================

# Your 2025 Brasil GP data
drivers_2025 = ["NOR", "ANT", "LEC", "PIA", "HAD", "RUS", "LAW", "BEA", 
                "GAS", "ALO", "HUL", "ALB", "HAM", "STR", "SAI", "VER", 
                "OCO", "COL", "TSU", "BOR"]

qualifying_times_2025 = [69.511, 69.685, 69.805, 69.886, 69.931, 69.942, 
                         69.962, 69.977, 70.002, 70.001, 70.039, 70.053, 
                         70.100, 70.161, 70.472, 70.403, 70.438, 70.632, 
                         70.711, 70.712]

brasil_2025 = pd.DataFrame({
    "Driver": drivers_2025,
    "QualifyingTime (s)": qualifying_times_2025
})

# Add historical averages for known drivers
brasil_2025["DriverHistoricalAvg (s)"] = brasil_2025["Driver"].map(driver_avg_times)

# For new drivers without history, use median
median_avg = training_data["DriverHistoricalAvg (s)"].median()
brasil_2025["DriverHistoricalAvg (s)"].fillna(median_avg, inplace=True)

# Add wet performance
brasil_2025["DriverWetAvg (s)"] = brasil_2025["Driver"].map(driver_wet_avg)
brasil_2025["DriverWetAvg (s)"].fillna(brasil_2025["DriverHistoricalAvg (s)"], inplace=True)

# Expected weather for race (you should update these with forecast)
brasil_2025["Temperature (°C)"] = 13 # Update with actual forecast
brasil_2025["Humidity (%)"] = 76    # Update with actual forecast
brasil_2025["Pressure (mbar)"] = 980 # Update with actual forecast
brasil_2025["Rainfall (mm)"] = 0      # Update with actual forecast

# Predict
X_brasil = brasil_2025[[
    "QualifyingTime (s)",
    "DriverHistoricalAvg (s)",
    "DriverWetAvg (s)",
    "Temperature (°C)",
    "Humidity (%)",
    "Pressure (mbar)",
    "Rainfall (mm)"
]]

brasil_2025["PredictedRaceTime (s)"] = model.predict(X_brasil)
brasil_2025 = brasil_2025.sort_values("PredictedRaceTime (s)")

print("\n🏁 Predicted 2025 Brasil GP Results 🏁\n")
print(brasil_2025[["Driver", "QualifyingTime (s)", "PredictedRaceTime (s)"]].to_string(index=False))

# ============================================
# STEP 5: Visualize feature importance
# ============================================

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color='#FF1801')
plt.xlabel("Importance")
plt.title("Feature Importance in F1 Race Time Prediction")
plt.tight_layout()
plt.show()