import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#GRAND_PRIX = 'Mexico City Grand Prix'
YEAR = 2025

FEATURES = [
    "Year",
    "AvgRaceTime (s)",
    "StartPosition",
    "DriverPointsBefore",
    "TeamPointsBefore",
]

TARGET = ["EndPosition"]

dataset = pd.read_csv(r"src\generated_data\training_data.csv")
#dataset.dropna()

training_ds = dataset[(dataset['Year'] < YEAR)].dropna()
test_ds = dataset[(dataset['Year'] == YEAR)].dropna()

X_train = training_ds[FEATURES]
X_test = test_ds[FEATURES]
y_train = training_ds[TARGET]
y_test = test_ds[TARGET]

#X = dataset[FEATURES]
#y = dataset[TARGET]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = RandomForestRegressor(
    
)

classifier.fit(X_train, y_train.values.ravel())  # .ravel() to avoid DataConversion warning

# Evaluate
y_pred = classifier.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Gradient Boosting Model')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()