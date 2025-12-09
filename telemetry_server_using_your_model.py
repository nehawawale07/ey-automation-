# telemetry_server_using_your_model.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn

app = FastAPI()

# load models (produced by train script)
rf = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")

FEATURE_ORDER = [
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

# per-vehicle rolling RPM state (simple in-memory)
rolling_states = {}

class Telemetry(BaseModel):
    vehicle_id: str = "vehicle_1"
    engine_rpm: float = 0.0
    coolant_temp: float = 0.0
    intake_temp: float = 0.0
    vehicle_speed: float = 0.0
    maf: float = 0.0
    map_kpa: float = 0.0
    throttle: float = 0.0

def compute_features(payload):
    vid = payload.get("vehicle_id", "vehicle_1")
    st = rolling_states.setdefault(vid, {"rpms":[]})
    rpms = st["rpms"]
    rpms.append(payload["engine_rpm"])
    if len(rpms) > 5:
        rpms.pop(0)
    rpm_mean = float(np.mean(rpms))
    rpm_std = float(np.std(rpms, ddof=0))
    row = {
        'Engine RPM [RPM]': payload["engine_rpm"],
        'rpm_rolling_mean': rpm_mean,
        'rpm_rolling_std': rpm_std,
        'Engine Coolant Temperature [°C]': payload["coolant_temp"],
        'Intake Air Temperature [°C]': payload["intake_temp"],
        'Vehicle Speed Sensor [km/h]': payload["vehicle_speed"],
        'Air Flow Rate from Mass Flow Sensor [g/s]': payload["maf"],
        'Intake Manifold Absolute Pressure [kPa]': payload["map_kpa"],
        'Absolute Throttle Position [%]': payload["throttle"]
    }
    # order columns and fill missing with 0
    row_df = pd.DataFrame([row])[FEATURE_ORDER].fillna(0)
    return row_df, row

@app.post("/telemetry")
def telemetry_endpoint(t: Telemetry):
    payload = t.dict()
    row_df, raw_feats = compute_features(payload)
    Xs = scaler.transform(row_df)
    prob = float(rf.predict_proba(Xs)[:,1])
    pred = int(prob > 0.5)
    health_score = round(100 - prob*60, 2)
    return {"prediction": pred, "probability": prob, "health_score": health_score, "features": raw_feats}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            row_df, raw_feats = compute_features(data)
            Xs = scaler.transform(row_df)
            prob = float(rf.predict_proba(Xs)[:,1])
            pred = int(prob > 0.5)
            health_score = round(100 - prob*60, 2)
            await ws.send_json({"prediction": pred, "probability": prob, "health_score": health_score, "features": raw_feats})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
