import time
import requests
import random
import json

# --- Configuration ---
# The URL of your FastAPI server (running on your network, replace with the server's actual IP)
SERVER_URL = "http://YOUR_SERVER_IP:8000/telemetry"
VEHICLE_ID = "RPI_Vehicle_101"

# Simulated starting values for data (to make the stream look realistic)
current_rpm = 1500.0
current_temp = 90.0
current_speed = 30.0

# --- Simulated Data Stream Function ---
def get_sensor_data():
    """Simulates reading current data from vehicle sensors."""
    global current_rpm, current_temp, current_speed
    
    # Introduce small, realistic fluctuations
    current_rpm += random.uniform(-50, 50)
    current_temp += random.uniform(-0.5, 0.5)
    current_speed += random.uniform(-2, 2)
    
    # Keep values within a reasonable range (e.g., clamp RPM)
    current_rpm = max(800.0, min(current_rpm, 4500.0))
    current_temp = max(80.0, min(current_temp, 110.0))
    current_speed = max(0.0, min(current_speed, 120.0))

    # Introduce a potential anomaly (optional: for testing high-temp fault)
    if random.random() < 0.05:
        current_temp += 10 # Sudden spike for 5% of readings

    # Data structure must match the FastAPI Telemetry Pydantic model
    payload = {
        "vehicle_id": VEHICLE_ID,
        "engine_rpm": round(current_rpm, 2),
        "coolant_temp": round(current_temp, 2),
        "intake_temp": round(random.uniform(25, 45), 2), # Simplified, static sensors
        "vehicle_speed": round(current_speed, 2),
        "maf": round(random.uniform(5, 50), 2),
        "map_kpa": round(random.uniform(50, 150), 2),
        "throttle": round(random.uniform(10, 80), 2)
    }
    return payload

# --- Core Logic: Send Data to API ---
def send_telemetry_to_server(data):
    """Sends the collected data payload to the server's API POST endpoint."""
    try:
        response = requests.post(SERVER_URL, json=data)
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
        
        # Output the response from the ML server
        result = response.json()
        
        print("\n--- SERVER RESPONSE ---")
        print(f"Prediction: {'FAULT DETECTED' if result['prediction'] == 1 else 'Normal'}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Health Score: {result['health_score']}%")
        print("-----------------------\n")
        
        # You would implement alert logic here (e.g., send email if prediction == 1)

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Failed to connect or send data to server: {e}")
        print(f"Attempted URL: {SERVER_URL}")

# --- Main Execution Loop (Simulating 1-Minute Interval) ---
if __name__ == "__main__":
    
    # --- STEP 3: Take 1 minute intervals of the data stream and store it ---
    # We are simulating that the server handles the rolling features, 
    # so the client just needs to send data consistently.
    
    # The original requirement was to collect for 1 minute THEN send.
    # A more robust real-time system sends data frequently (e.g., every 5 seconds)
    # and lets the server handle the rolling window for features. We'll use the frequent approach.
    
    # Set the desired interval for sending data (e.g., every 5 seconds)
    SEND_INTERVAL_SECONDS = 5 
    
    print(f"Starting Raspberry Pi data stream client. Sending data every {SEND_INTERVAL_SECONDS} seconds.")
    print(f"Targeting server: {SERVER_URL}")

    while True:
        try:
            # 1. Get new data from the "sensors"
            telemetry_payload = get_sensor_data()
            print(f"Sending data for RPM: {telemetry_payload['engine_rpm']} / Temp: {telemetry_payload['coolant_temp']}")
            
            # 2. Send data to the server API POST endpoint
            # --- STEP 4 & 5: Pass data to ML model (server handles this) & output goes to API POST ---
            send_telemetry_to_server(telemetry_payload)
            
            # Wait for the next send interval
            time.sleep(SEND_INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            print("\nClient stopped by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(SEND_INTERVAL_SECONDS)