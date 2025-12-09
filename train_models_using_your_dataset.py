# train_models_using_your_dataset.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle  # Changed from joblib

df = pd.read_csv("vehicle_data_from_your_csvs.csv")  # Changed from parquet to csv

features = [
    'Engine RPM [RPM]',
    'rpm_rolling_mean',
    'rpm_rolling_std',
    'Engine Coolant Temperature [°C]',
    'Intake Air Temperature [°C]',
    'Vehicle Speed Sensor [km/h]',
    'Air Flow Rate from Mass Flow Sensor [g/s]',
    'Intake Manifold Absolute Pressure [kPa]',
    'Absolute Throttle Position [%]'
]
features = [f for f in features if f in df.columns]

X = df[features].fillna(0)
y = df['problem_flag'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train_s, y_train)

y_pred = rf.predict(X_test_s)
print("Classification report (RandomForest):")
print(classification_report(y_test, y_pred))
try:
    print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_s)[:,1]))
except Exception:
    pass

# Save with pickle (compatible with joblib-loaded files too)
with open("rf_model.joblib", "wb") as f:
    pickle.dump(rf, f)
with open("scaler.joblib", "wb") as f:
    pickle.dump(scaler, f)
print("Saved rf_model.joblib and scaler.joblib")
