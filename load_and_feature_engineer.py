{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d7cd07ff",
   "metadata": {},
   "source": [
    "Loads the  CSV files , cleans columns, computes rolling features and the detection flags (overheating, high RPM, etc.)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1bcf3754",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load_and_feature_engineer.py\n",
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4eb1561b",
   "metadata": {},
   "outputs": [],
   "source": [
    "file_urls = [\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-05_Seat_Leon_RT_S_Stau.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-05_Seat_Leon_S_KA_Normal.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-06_Seat_Leon_KA_KA_Normal.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-06_Seat_Leon_KA_RT_Normal.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-07_Seat_Leon_S_RT_Normal.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-10_Seat_Leon_KA_KA_Stau.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-11_Seat_Leon_KA_KA_Stau.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-11_Seat_Leon_KA_S_Normal.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-11_Seat_Leon_S_RT_Frei.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-12_Seat_Leon_RT_S_Normal.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-12_Seat_Leon_S_RT_Normal.csv\",\n",
    "    \"https://raw.githubusercontent.com/hayatu4islam/Automotive_Diagnostics/main/OBD-II-Dataset/2017-07-13_Seat_Leon_KA_KA_Normal.csv\"\n",
    "]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "05b8fe3d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading CSVs\n",
      "Loaded rows: 368341\n"
     ]
    }
   ],
   "source": [
    "print(\"Downloading CSVs\")\n",
    "dfs = [pd.read_csv(url) for url in file_urls]\n",
    "df = pd.concat(dfs, ignore_index=True)\n",
    "print(\"Loaded rows:\", len(df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "357ea72e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean column names \n",
    "df.columns = df.columns.str.replace('Â', '', regex=False).str.strip()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5396748f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipython-input-147625850.py:12: FutureWarning: errors='ignore' is deprecated and will raise in a future version. Use to_numeric without passing `errors` and catch exceptions explicitly instead\n",
      "  df[col] = pd.to_numeric(df[col], errors='ignore')\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# For reproducibility — ensure expected columns exist (rename if necessary)\n",
    "# Notebook used columns like:\n",
    "# 'Engine Coolant Temperature [°C]', 'Intake Manifold Absolute Pressure [kPa]',\n",
    "# 'Engine RPM [RPM]', 'Vehicle Speed Sensor [km/h]', 'Intake Air Temperature [°C]',\n",
    "# 'Air Flow Rate from Mass Flow Sensor [g/s]', 'Absolute Throttle Position [%]'\n",
    "# If your dataset uses slightly different names, you can map them here\n",
    "\n",
    "# Convert types where possible\n",
    "for col in df.columns:\n",
    "    # try to coerce numeric columns to numeric\n",
    "    try:\n",
    "        df[col] = pd.to_numeric(df[col], errors='ignore')\n",
    "    except Exception:\n",
    "        pass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "96848e8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clip coolant temps \n",
    "coolant_col = \"Engine Coolant Temperature [°C]\"\n",
    "if coolant_col in df.columns:\n",
    "    df[coolant_col] = df[coolant_col].clip(60, 130)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bf331ca6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compute rolling RPM features \n",
    "rpm_col = \"Engine RPM [RPM]\"\n",
    "if rpm_col in df.columns:\n",
    "    # ensure numeric and fillna to avoid rolling issues\n",
    "    df[rpm_col] = pd.to_numeric(df[rpm_col], errors='coerce').fillna(0).astype(float)\n",
    "    df['rpm_rolling_mean'] = df[rpm_col].rolling(window=5, min_periods=1).mean()\n",
    "    df['rpm_rolling_std'] = df[rpm_col].rolling(window=5, min_periods=1).std().fillna(0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5116a345",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature engineering & detection rules \n",
    "iat_col = \"Intake Air Temperature [°C]\"\n",
    "speed_col = \"Vehicle Speed Sensor [km/h]\"\n",
    "maf_col = \"Air Flow Rate from Mass Flow Sensor [g/s]\"\n",
    "map_col = \"Intake Manifold Absolute Pressure [kPa]\"\n",
    "throttle_col = \"Absolute Throttle Position [%]\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ad90d198",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill NA for numeric cols so boolean ops don't fail\n",
    "for c in [coolant_col, iat_col, speed_col, maf_col, map_col, throttle_col]:\n",
    "    if c in df.columns:\n",
    "        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c1c31745",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Detection flags\n",
    "df['high_rpm'] = (df[rpm_col] > 3000).astype(int) if rpm_col in df.columns else 0\n",
    "df['engine_overheat'] = ((df[coolant_col] > 100) | (df[iat_col] > 80)).astype(int) if coolant_col in df.columns and iat_col in df.columns else 0\n",
    "df['overspeed'] = (df[speed_col] > 120).astype(int) if speed_col in df.columns else 0\n",
    "df['throttle_abuse'] = ((df[throttle_col] > 70) & (df[speed_col] < 30)).astype(int) if throttle_col in df.columns and speed_col in df.columns else 0\n",
    "df['low_efficiency'] = ((df[maf_col] > 40) & (df[speed_col] < 25)).astype(int) if maf_col in df.columns and speed_col in df.columns else 0\n",
    "df['pressure_fault'] = ((df[map_col] > 180) | (df[map_col] < 50)).astype(int) if map_col in df.columns else 0\n",
    "df['rpm_anomaly'] = (df['rpm_rolling_std'] > 350).astype(int) if 'rpm_rolling_std' in df.columns else 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "44c221c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Master problem flag and health score (same scoring)\n",
    "df['problem_flag'] = (\n",
    "    df['high_rpm'] |\n",
    "    df['engine_overheat'] |\n",
    "    df['overspeed'] |\n",
    "    df['throttle_abuse'] |\n",
    "    df['low_efficiency'] |\n",
    "    df['pressure_fault'] |\n",
    "    df['rpm_anomaly']\n",
    ").astype(int)\n",
    "\n",
    "df['vehicle_health_score'] = 100 - (\n",
    "    df['high_rpm'] * 5 +\n",
    "    df['engine_overheat'] * 20 +\n",
    "    df['overspeed'] * 10 +\n",
    "    df['throttle_abuse'] * 8 +\n",
    "    df['low_efficiency'] * 15 +\n",
    "    df['pressure_fault'] * 10 +\n",
    "    df['rpm_anomaly'] * 12\n",
    ")\n",
    "df['vehicle_health_score'] = df['vehicle_health_score'].clip(0,100)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b66de391",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saved: vehicle_data_from_your_csvs.parquet\n",
      "   Engine RPM [RPM]  rpm_rolling_mean  rpm_rolling_std  \\\n",
      "0               0.0               0.0              0.0   \n",
      "1               0.0               0.0              0.0   \n",
      "2               0.0               0.0              0.0   \n",
      "3               0.0               0.0              0.0   \n",
      "4               0.0               0.0              0.0   \n",
      "\n",
      "   Engine Coolant Temperature [°C]  Intake Air Temperature [°C]  \\\n",
      "0                               60                          0.0   \n",
      "1                               60                          0.0   \n",
      "2                               60                          0.0   \n",
      "3                               60                          0.0   \n",
      "4                               60                         22.0   \n",
      "\n",
      "   Vehicle Speed Sensor [km/h]  Air Flow Rate from Mass Flow Sensor [g/s]  \\\n",
      "0                          0.0                                        0.0   \n",
      "1                          0.0                                        0.0   \n",
      "2                          0.0                                        0.0   \n",
      "3                          0.0                                        0.0   \n",
      "4                          0.0                                        0.0   \n",
      "\n",
      "   Absolute Throttle Position [%]  problem_flag  vehicle_health_score  \n",
      "0                             0.0             1                    90  \n",
      "1                             0.0             0                   100  \n",
      "2                             0.0             0                   100  \n",
      "3                             0.0             0                   100  \n",
      "4                             0.0             0                   100  \n"
     ]
    }
   ],
   "source": [
    "# Save the cleaned + engineered dataset (use same filename your pipeline expects)\n",
    "df.to_parquet(\"vehicle_data_from_your_csvs.parquet\", index=False)\n",
    "\n",
    "print(\"Saved: vehicle_data_from_your_csvs.parquet\")\n",
    "print(df[['Engine RPM [RPM]', 'rpm_rolling_mean','rpm_rolling_std',\n",
    "          'Engine Coolant Temperature [°C]','Intake Air Temperature [°C]',\n",
    "          'Vehicle Speed Sensor [km/h]','Air Flow Rate from Mass Flow Sensor [g/s]',\n",
    "          'Absolute Throttle Position [%]','problem_flag','vehicle_health_score']].head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76522c1c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
