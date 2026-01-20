from fastapi import FastAPI, Query, HTTPException
from backend import api_utils
import numpy as np
import pandas as pd
import torch
from backend import models
from pathlib import Path

app = FastAPI(title="Rockfall Early Warning API")


DATA_FILES = [
    r"C:\Users\HP\Downloads\AE_Damage_Detection_Dataset.csv",
    r"C:\Users\HP\Downloads\excavation_risk_dataset.csv",
    r"C:\Users\HP\Downloads\hudsonmt.out\hudsonmt.out" 
]


FEATURE_COLS = [
    "vibration_energy",
    "vibration_entropy",
    "acoustic_energy",
    "acoustic_hit_rate",
    "stress_index",
    "excavation_depth",
    "base_geotech_risk",
    "soil_type"
]


lstm_model = models.RockfallLSTM(input_size=len(FEATURE_COLS))
ae_model = models.RockfallAutoencoder(input_size=len(FEATURE_COLS))

lstm_model.eval()
ae_model.eval()

def load_all_data(feature_cols: list):
    """Load all CSV/OUT files and combine into a single DataFrame"""
    dfs = []
    for file in DATA_FILES:
        path = Path(file)
        if not path.exists():
            print(f"⚠️ Warning: file not found -> {file}")
            continue
        try:
            if path.suffix == ".out":
                df = pd.read_csv(path, delim_whitespace=True, parse_dates=["timestamp"])
            else:
                df = pd.read_csv(path, parse_dates=["timestamp"])
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Warning reading {file}: {e}")

    if not dfs:
        raise HTTPException(status_code=500, detail="No data files loaded!")

    data = pd.concat(dfs, ignore_index=True)

    
    if "soil_type" in feature_cols and "soil_type" in data.columns:
        data["soil_type"] = data["soil_type"].astype("category").cat.codes

    return data


def load_latest_zone_features(data: pd.DataFrame, zone_id: str):
    """Load latest features for a given zone"""
    zone_df = data[data["zone_id"] == zone_id]
    if zone_df.empty:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")

    latest_row = zone_df.sort_values("timestamp").iloc[-1]
    features = latest_row[FEATURE_COLS].values.reshape(1, -1)
    return features, latest_row


def compute_risk(features: np.ndarray):
    """Compute risk score using LSTM + Autoencoder"""
    return api_utils.predict_risk(features, FEATURE_COLS, lstm_model, ae_model)


def get_risk_label(score: float):
    """Return Safe / Warning / Danger"""
    return api_utils.risk_category(score)



data = load_all_data(FEATURE_COLS)



@app.get("/zones")
def list_zones():
    """List all zones in the dataset"""
    return sorted(data["zone_id"].unique().tolist())


@app.get("/predict-risk")
def predict_risk_endpoint(zone: str = Query(..., description="Zone ID from dataset")):
    """Predict risk for a single zone"""
    features, meta = load_latest_zone_features(data, zone)
    score = compute_risk(features)
    label = get_risk_label(score)
    explanation = api_utils.get_explanation(features, FEATURE_COLS)
    return {
        "zone": zone,
        "timestamp": str(meta["timestamp"]),
        "risk_score": round(float(score), 4),
        "risk_label": label,
        "explanation": explanation
    }


@app.get("/zone-status")
def zone_status():
    """Predict risk for all zones"""
    response = []
    for zone in data["zone_id"].unique():
        try:
            features, meta = load_latest_zone_features(data, zone)
            score = compute_risk(features)
            label = get_risk_label(score)
            response.append({
                "zone": zone,
                "timestamp": str(meta["timestamp"]),
                "risk_score": round(float(score), 4),
                "risk_label": label
            })
        except Exception as e:
            response.append({
                "zone": zone,
                "error": str(e)
            })
    return response


@app.get("/get-explanation")
def get_explanation_endpoint(zone: str = Query(...)):
    """Get explanation for a single zone"""
    features, _ = load_latest_zone_features(data, zone)
    explanation = api_utils.get_explanation(features, FEATURE_COLS)
    return {
        "zone": zone,
        "explanation": explanation
    }
