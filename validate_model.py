# validate_model.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import pickle  # Changed from joblib

# Load trained model + data
df = pd.read_csv("vehicle_data_from_your_csvs.csv")  # Changed from parquet to csv
rf = pickle.load(open("rf_model.joblib", "rb"))  # Changed from joblib
scaler = pickle.load(open("scaler.joblib", "rb"))  # Changed from joblib

FEATURES = [
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
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES].fillna(0)
y = df["problem_flag"]

Xs = scaler.transform(X)
pred = rf.predict(Xs)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y, pred))

try:
    auc = roc_auc_score(y, rf.predict_proba(Xs)[:,1])
    print("ROC AUC:", auc)
except:
    pass

cm = confusion_matrix(y, pred)
print("\nCONFUSION MATRIX:")
print(cm)

# plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Vehicle Fault Detection Confusion Matrix")
plt.show()
