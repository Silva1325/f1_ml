import fastf1
import pandas as pd
import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt
import fastf1.plotting

# Setup plotting
fastf1.plotting.setup_mpl(color_scheme='fastf1', misc_mpl_mods=False)

# Load a specific race session (e.g., 2025 Spanish GP Race)
year = 2025
gp = 'Spain'  # Or any track name/ID
session_type = 'R'  # 'R' for Race
session = fastf1.get_session(year, gp, session_type)
session.load()  # Loads laps, telemetry, etc.

# Get lap data
laps = session.laps

# Clean laps: Accurate, no NaN times, no pit laps, clear track status
laps_cleaned = laps.loc[
    (laps['IsAccurate'] == True) &
    (laps['LapTime'].notna()) &
    (laps['PitOutTime'].isna()) &
    (laps['PitInTime'].isna()) &
    (laps['TrackStatus'] == '1')  # '1' means clear track
].copy()

# Convert lap times to seconds for analysis
laps_cleaned['LapTimeSeconds'] = laps_cleaned['LapTime'].dt.total_seconds()

# Get unique drivers
drivers = pd.unique(laps_cleaned['Driver'])

# Prepare plot and results storage
fig, ax = plt.subplots(figsize=(20, 10))
compound_colors = {
    compound: fastf1.plotting.get_compound_color(compound, session=session)
    for compound in laps_cleaned['Compound'].unique()
}
degradation_data = []  # For summary table

# Analyze per driver and stint
for driver in drivers:
    driver_laps = laps_cleaned.loc[laps_cleaned['Driver'] == driver]
    stints = driver_laps['Stint'].unique()
    
    for stint_number in stints:
        stint_laps = driver_laps.loc[driver_laps['Stint'] == stint_number].copy()
        
        if len(stint_laps) < 3:  # Need at least 3 laps for reliable regression
            continue
        
        compound = stint_laps['Compound'].iloc[0]
        
        # Calculate tyre age (laps in stint, starting at 1)
        stint_laps['StintLapNumber'] = stint_laps['LapNumber'] - stint_laps['LapNumber'].min() + 1
        
        # Linear regression: Lap time vs. tyre age
        slope, intercept, r_value, p_value, std_err = linregress(
            stint_laps['StintLapNumber'], stint_laps['LapTimeSeconds']
        )
        
        # Degradation rate in ms/lap (positive slope means slowing down)
        degradation_rate = slope * 1000  # Convert seconds to ms
        
        # Store results
        degradation_data.append({
            'Driver': driver,
            'Stint': stint_number,
            'Compound': compound,
            'Degradation (ms/lap)': round(degradation_rate, 1),
            'R²': round(r_value**2, 2)  # Fit quality
        })
        
        # Plot data points and regression line
        ax.plot(stint_laps['StintLapNumber'], stint_laps['LapTimeSeconds'],
                marker='o', linestyle='-', color=compound_colors[compound], label=f"{driver} Stint {stint_number}")
        ax.plot(stint_laps['StintLapNumber'],
                intercept + slope * stint_laps['StintLapNumber'],
                linestyle='--', color=compound_colors[compound], alpha=0.6)

# Finalize plot
ax.set_title(f'Tyre Degradation Analysis - {gp} GP {year}')
ax.set_xlabel('Laps in Stint')
ax.set_ylabel('Lap Time (seconds)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
plt.tight_layout()
plt.show()

# Display summary table
degradation_df = pd.DataFrame(degradation_data)
print(degradation_df)