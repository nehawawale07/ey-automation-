# Vehicle Telemetry Anomaly Detection Pipeline

Complete ML pipeline for **real-time vehicle telemetry anomaly detection** using Raspberry Pi hardware integration.

## 🚀 Quick Start (Hardware Team)

Execute files in **EXACT order** below. Each step builds on previous outputs.

### 1. Environment Setup
git clone https://github.com/nehawawale07/ey-automation-.git
cd ey-automation-
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install pandas scikit-learn numpy fastapi uvicorn joblib paho-mqtt requests


### 2. Data Preparation (Run Once)
python load_and_feature_engineer.py

**What it does**: Loads `anomaly.csv` → Creates **CSV-based** feature-engineered dataset → Ready for ML training [attached_file:1]

### 3. Train ML Models (Run Once or Retrain)
python train_models_using_your_dataset.py

**What it does**: Trains anomaly detection models on CSV data → Saves `model.pkl` → Production-ready ML artifacts [attached_file:1]

### 4. Validate Models
python validate_model.py

**What it does**: Tests trained models on validation CSV data → Reports accuracy metrics → Confirms model quality [attached_file:1]

### 5. Start Telemetry Server (Production)
python telemetry_server_using_your_model.py

**What it does**: 
- Loads trained `model.pkl`
- Starts FastAPI server on `0.0.0.0:8000`
- Receives real-time vehicle telemetry via HTTP/MQTT
- Returns anomaly predictions [attached_file:1]

### 6. Raspberry Pi Hardware Client
python rpi_data_client.py

**What it does**: 
- Connects to vehicle CAN bus/sensors
- Streams telemetry data to server (step 5)
- Displays real-time anomaly alerts [attached_file:1]

### 7. Test Data Stream (Optional)
python csv_stream_simulator.py

**What it does**: Simulates vehicle data stream from CSV → Tests end-to-end pipeline without hardware [attached_file:1]

## 📋 File Execution Order
load_and_feature_engineer.py → Processes anomaly.csv

train_models_using_your_dataset.py → Creates model.pkl

validate_model.py → Confirms model works

telemetry_server_using_your_model.py → Start server

rpi_data_client.py → Hardware streaming


## 🛠️ Hardware Integration
Raspberry Pi → rpi_data_client.py → Telemetry Server → ML Model → Anomaly Alerts



## ✅ Expected Outputs
- Processed **CSV datasets** (from anomaly.csv)
- `model.pkl` (trained model)
- FastAPI server running on `:8000`
- Real-time anomaly predictions

## 🔍 API Endpoints (After Step 5)
POST /predict → {"accelerometer": 1.2, "gyroscope": 0.8, ...} → {"anomaly": true, "confidence": 0.92}


**Note**: Using **CSV format only** (no parquet files required)

---
*Built for EY Automation - Production Ready Pipeline*

