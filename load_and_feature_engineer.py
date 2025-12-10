# load_and_feature_engineer.py

import pandas as pd
import numpy as np

file_urls = [
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-05_Seat_Leon_RT_S_Stau.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-05_Seat_Leon_S_KA_Normal.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-06_Seat_Leon_KA_KA_Normal.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-06_Seat_Leon_KA_RT_Normal.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-07_Seat_Leon_S_RT_Normal.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-10_Seat_Leon_KA_KA_Stau.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-11_Seat_Leon_KA_KA_Stau.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-11_Seat_Leon_KA_S_Normal.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-11_Seat_Leon_S_RT_Frei.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-12_Seat_Leon_RT_S_Normal.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-12_Seat_Leon_S_RT_Normal.csv",
    "https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-13_Seat_Leon_KA_KA_Normal.csv"
]

print("Downloading CSVs")
dfs = [pd.read_csv(url) for url in file_urls]
df = pd.concat(dfs, ignore_index=True)
print("Loaded rows:", len(df))

# Clean column names
df.columns = df.columns.str.replace('Â', '', regex=False).str.strip()

# Convert columns to numeric where possible
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    except Exception:
        pass

# Clip coolant temps to range [60, 130]
coolant_col = "Engine Coolant Temperature [°C]"
if coolant_col in df.columns:
    df[coolant_col] = df[coolant_col].clip(60, 130)

# Compute rolling RPM features
rpm_col = "Engine RPM [RPM]"
if rpm_col in df.columns:
    df[rpm_col] = pd.to_numeric(df[rpm_col], errors='coerce').fillna(0).astype(float)
    df['rpm_rolling_mean'] = df[rpm_col].rolling(window=5, min_periods=1).mean()
    df['rpm_rolling_std'] = df[rpm_col].rolling(window=5, min_periods=1).std().fillna(0)

# Feature engineering and detection columns
iat_col = "Intake Air Temperature [°C]"
speed_col = "Vehicle Speed Sensor [km/h]"
maf_col = "Air Flow Rate from Mass Flow Sensor [g/s]"
map_col = "Intake Manifold Absolute Pressure [kPa]"
throttle_col = "Absolute Throttle Position [%]"

# Fill NA in important numeric columns
for c in [coolant_col, iat_col, speed_col, maf_col, map_col, throttle_col]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# Detection flags
df['high_rpm'] = (df[rpm_col] > 3000).astype(int) if rpm_col in df.columns else 0
df['engine_overheat'] = ((df[coolant_col] > 100) | (df[iat_col] > 80)).astype(int) if coolant_col in df.columns and iat_col in df.columns else 0
df['overspeed'] = (df[speed_col] > 120).astype(int) if speed_col in df.columns else 0
df['throttle_abuse'] = ((df[throttle_col] > 70) & (df[speed_col] < 30)).astype(int) if throttle_col in df.columns and speed_col in df.columns else 0
df['low_efficiency'] = ((df[maf_col] > 40) & (df[speed_col] < 25)).astype(int) if maf_col in df.columns and speed_col in df.columns else 0
df['pressure_fault'] = ((df[map_col] > 180) | (df[map_col] < 50)).astype(int) if map_col in df.columns else 0
df['rpm_anomaly'] = (df['rpm_rolling_std'] > 350).astype(int) if 'rpm_rolling_std' in df.columns else 0

# Master problem flag and health score
df['problem_flag'] = (
    df['high_rpm'] |
    df['engine_overheat'] |
    df['overspeed'] |
    df['throttle_abuse'] |
    df['low_efficiency'] |
    df['pressure_fault'] |
    df['rpm_anomaly']
).astype(int)

df['vehicle_health_score'] = 100 - (
    df['high_rpm'] * 5 +
    df['engine_overheat'] * 20 +
    df['overspeed'] * 10 +
    df['throttle_abuse'] * 8 +
    df['low_efficiency'] * 15 +
    df['pressure_fault'] * 10 +
    df['rpm_anomaly'] * 12
)

df['vehicle_health_score'] = df['vehicle_health_score'].clip(0, 100)

# Save the cleaned and engineered dataset
df.to_parquet("vehicle_data_from_your_csvs.parquet", index=False)
print("Saved: vehicle_data_from_your_csvs.parquet")

print(df[['Engine RPM [RPM]', 'rpm_rolling_mean', 'rpm_rolling_std',
          'Engine Coolant Temperature [°C]', 'Intake Air Temperature [°C]',
          'Vehicle Speed Sensor [km/h]', 'Air Flow Rate from Mass Flow Sensor [g/s]',
          'Absolute Throttle Position [%]', 'problem_flag', 'vehicle_health_score']].head())
