
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

YEAR = 2025
PREVIOUS_YEAR = YEAR - 1

PREVIOUS_GRAND_PRIX_PRIX = "Mexico City Grand Prix"
GRAND_PRIX = "São Paulo Grand Prix"

dataset = pd.read_csv(r"src\generated_data\races_data.csv")
#previous_years_gp_ds = dataset[(dataset["GP"] == GRAND_PRIX)]
previous_years_gp_ds = dataset[(dataset["Year"] == PREVIOUS_YEAR) & (dataset["GP"] == GRAND_PRIX)]

# Indepentend variables and dependent variable
X = previous_years_gp_ds[
    [
        "QualifyingTime (s)",
        "Temperature (°C)",
        "Humidity (%)",
        "Pressure (mbar)",
        "Rainfall (mm)",
        "TrackTemp (°C)",
        "DriverHistoricalAvg (s)",
        "Driver_Dry_Perf",
        "Driver_Damp_Perf",
        "Driver_Wet_Perf",
        "Driver_Cold_Perf",
        "Driver_Mid_Perf",
        "Driver_Hot_Perf",
        "Driver_HighDownforce_Perf",
        "Driver_LowDownforce_Perf",
        "DriverEloBefore",
        "ConstructorEloBefore"
    ]
].values
y = previous_years_gp_ds["AvgRaceTime (s)"].values

# Splitting the dataset into the Trainning Set and Test Set
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Trainning the Decision Tree model into the whole dataset 
classifier = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
classifier.fit(X_train,y_train)

# Evaluate
y_pred = classifier.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(mae)

# Predict values
drivers_2025 = ["NOR", "ANT", "LEC", "PIA", "HAD", "RUS", "LAW", "BEA", 
                "GAS", "ALO", "HUL", "ALB", "HAM", "STR", "SAI", "VER", 
                "OCO", "COL", "TSU", "BOR"]

qualifying_times_2025 = [69.511, 69.685, 69.805, 69.886, 69.931, 69.942, 
                         69.962, 69.977, 70.002, 70.001, 70.039, 70.053, 
                         70.100, 70.161, 70.472, 70.403, 70.438, 70.632, 
                         70.711, 70.712]

last_gp_data = dataset[
    (dataset["GP"] == PREVIOUS_GRAND_PRIX_PRIX) &
    (dataset["Year"] == YEAR) &
    (dataset["Driver"].isin(drivers_2025))
]

print((last_gp_data.set_index("Driver").loc[drivers_2025, "DriverHistoricalAvg (s)"].values))

brasil_2025 = pd.DataFrame({
    "Driver": drivers_2025,
    "QualifyingTime (s)": qualifying_times_2025,
    "DriverHistoricalAvg (s)": (last_gp_data.set_index("Driver").loc[drivers_2025, "DriverHistoricalAvg (s)"].values),
    "Driver_Dry_Perf": (last_gp_data.set_index("Driver").loc[drivers_2025, "Driver_Dry_Perf"].values),
    "Driver_Damp_Perf": (last_gp_data.set_index("Driver").loc[drivers_2025, "Driver_Damp_Perf"].values),
    "Driver_Wet_Perf": (last_gp_data.set_index("Driver").loc[drivers_2025, "Driver_Wet_Perf"].values),
    "Driver_Cold_Perf": (last_gp_data.set_index("Driver").loc[drivers_2025, "Driver_Cold_Perf"].values),
    "Driver_Mid_Perf": (last_gp_data.set_index("Driver").loc[drivers_2025, "Driver_Mid_Perf"].values),
    "Driver_Hot_Perf": (last_gp_data.set_index("Driver").loc[drivers_2025, "Driver_Hot_Perf"].values),
    "Driver_HighDownforce_Perf": (last_gp_data.set_index("Driver").loc[drivers_2025, "Driver_HighDownforce_Perf"].values),
    "Driver_LowDownforce_Perf": (last_gp_data.set_index("Driver").loc[drivers_2025, "Driver_LowDownforce_Perf"].values),
    "DriverEloBefore": (last_gp_data.set_index("Driver").loc[drivers_2025, "DriverEloBefore"].values),
    "ConstructorEloBefore": (last_gp_data.set_index("Driver").loc[drivers_2025, "ConstructorEloBefore"].values),
})

brasil_2025["Temperature (°C)"] = 18.7 # Update with actual forecast
brasil_2025["Humidity (%)"] = 72    # Update with actual forecast
brasil_2025["Pressure (mbar)"] = 980 # Update with actual forecast
brasil_2025["Rainfall (mm)"] = 0.2      # Update with actual forecast
brasil_2025["TrackTemp (°C)"] = 20.1      # Update with actual forecast


# Predict
X_brasil = brasil_2025[[
    "QualifyingTime (s)",
    "Temperature (°C)",
    "Humidity (%)",
    "Pressure (mbar)",
    "Rainfall (mm)",
    "TrackTemp (°C)",
    "DriverHistoricalAvg (s)",
    "Driver_Dry_Perf",
    "Driver_Damp_Perf",
    "Driver_Wet_Perf",
    "Driver_Cold_Perf",
    "Driver_Mid_Perf",
    "Driver_Hot_Perf",
    "Driver_HighDownforce_Perf",
    "Driver_LowDownforce_Perf",
    "DriverEloBefore",
    "ConstructorEloBefore",
]]

brasil_2025["PredictedRaceTime (s)"] = classifier.predict(X_brasil)
brasil_2025 = brasil_2025.sort_values("PredictedRaceTime (s)")

print("\n🏁 Predicted 2025 Brasil GP Results 🏁\n")
print(brasil_2025[["Driver", "QualifyingTime (s)", "PredictedRaceTime (s)"]].to_string(index=False))