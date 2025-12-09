# csv_stream_simulator.py


import pandas as pd
import requests
import time

URL = "http://127.0.0.1:8000/telemetry"
DATA = pd.read_csv("anomaly.csv")  # Changed from parquet to csv

print("Starting realtime dataset stream...")

for idx, row in DATA.iterrows():

    payload = {
        "vehicle_id":"sim_car_001",
        "engine_rpm": float(row["Engine RPM [RPM]"]),
        "coolant_temp": float(row["Engine Coolant Temperature [°C]"]),
        "intake_temp": float(row["Intake Air Temperature [°C]"]),
        "vehicle_speed": float(row["Vehicle Speed Sensor [km/h]"]),
        "maf": float(row["Air Flow Rate from Mass Flow Sensor [g/s]"]),
        "map_kpa": float(row["Intake Manifold Absolute Pressure [kPa]"]),
        "throttle": float(row["Absolute Throttle Position [%]"])
    }

    r = requests.post(URL, json=payload)

    if r.status_code == 200:
        result = r.json()
        print(
            f"Health={result['health_score']} | "
            f"Prob={round(result['probability'],3)} | "
            f"Fault={bool(result['prediction'])}"
        )

    time.sleep(0.5)   # simulate real driving sensor delay
