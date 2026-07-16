import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load data from data folder
print("Loading data...")
df = pd.read_csv('data/cleaned_aqi.csv')  # <-- FIXED PATH

# Check if already clean (no NaN)
print(f"Data shape: {df.shape}")
print(f"Any NaN values: {df.isnull().sum().sum()}")

# Features and target
num_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
X = df[num_cols]
y = df['AQI']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\n" + "="*40)
print("MODEL PERFORMANCE")
print("="*40)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print("="*40)

# Save model
joblib.dump(model, 'aqi_model.pkl')
print("\nModel saved as 'aqi_model.pkl'")