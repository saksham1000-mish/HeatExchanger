import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your dataset (update path if needed)
df = pd.read_csv("RareTechSEA1-60.csv")

# Optional: replace these with actual mapping if needed
fluid_mapping = {"Water": 0, "Oil": 1, "Refrigerant": 2}
df["Fluid1"] = df["Fluid1"].map(fluid_mapping)
df["Fluid2"] = df["Fluid2"].map(fluid_mapping)

# List of final input features to match Streamlit
features = [
    "Thickness", "Plate_thickness_[mm]", "Temp1In", "Temp1Out", "Temp2In", "Temp2Out",
    "Fluid1", "Fluid2", "Q", "Qr", "Lmtd", "Theta1", "Theta2",
    "m1", "m2", "dP1", "dP2", "V1In", "V2In", "A", "NP", "AdNP", "NuP", "Ar"
]

# Target column (created earlier)
target = "maintenance_required"

# Drop rows with missing values (optional)
df = df[features + [target]].dropna()

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved to model.pkl")
