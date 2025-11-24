from matplotlib import pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_absolute_error

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

model = xgb.XGBRegressor()

model.fit(X_train,y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
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