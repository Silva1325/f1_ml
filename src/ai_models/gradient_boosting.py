
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#GRAND_PRIX = 'Mexico City Grand Prix'
YEAR = 2025

FEATURES = [
    "AvgRaceTime (s)",
    "StartPosition",
    "DriverPointsBefore",
    "TeamPointsBefore",
]

TARGET = ["EndPosition"]

dataset = pd.read_csv(r"src\generated_data\training_data.csv")

training_ds = dataset[(dataset['Year'] < YEAR)].dropna()
test_ds = dataset[(dataset['Year'] == YEAR)].dropna()

X_train = training_ds[FEATURES]
X_test = test_ds[FEATURES]
y_train = training_ds[TARGET]
y_test = test_ds[TARGET]

classifier = GradientBoostingRegressor()

classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# After training the model, add this code:

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
