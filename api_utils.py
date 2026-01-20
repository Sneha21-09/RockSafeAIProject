import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "backend"

SCALER_PATH = MODEL_DIR / "scaler.joblib"
LOOKBACK = 10  


if not SCALER_PATH.exists():
    raise FileNotFoundError("❌ Scaler missing. Please check your /models folder.")

scaler = joblib.load(SCALER_PATH)


class RockfallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  
        return self.sigmoid(out)


class RockfallAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_repeated = h[-1].unsqueeze(0).repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(h_repeated)
        return out


def load_all_data(feature_cols: list):
    """
    Load all CSVs and OUT files into a single DataFrame
    """
    csv_files = [
        DATA_DIR / "AE_Damage_Detection_Dataset.csv",
        DATA_DIR / "excavation_risk_dataset.csv",
        DATA_DIR / "hudsonmt.out"
    ]

    dfs = []
    for file in csv_files:
        try:
            if not file.exists():
                continue
            if file.suffix == ".out":
                df = pd.read_csv(file, delim_whitespace=True, parse_dates=["timestamp"])
            else:
                df = pd.read_csv(file, parse_dates=["timestamp"])
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Warning reading {file}: {e}")

    if not dfs:
        raise FileNotFoundError("No CSV data loaded!")

    data = pd.concat(dfs, ignore_index=True)

    if "soil_type" in feature_cols and "soil_type" in data.columns:
        data["soil_type"] = data["soil_type"].astype("category").cat.codes

    return data


def load_latest_features_zone(data: pd.DataFrame, zone_id: str, feature_cols: list):
    """
    Extract last LOOKBACK rows for a zone
    """
    zone_df = data[data["zone_id"] == zone_id].sort_values("timestamp")

    if len(zone_df) < LOOKBACK:
        raise ValueError(f"Zone {zone_id} needs at least {LOOKBACK} rows for LSTM prediction.")

    latest_sequence = zone_df.tail(LOOKBACK)
    features = latest_sequence[feature_cols].values.reshape(1, LOOKBACK, -1)
    latest_row = latest_sequence.iloc[-1]

    return features, latest_row



def predict_risk(features_seq: np.ndarray, feature_names: list,
                 lstm_model: nn.Module, ae_model: nn.Module) -> float:
    s_batch, s_len, s_feat = features_seq.shape
    flat_data = features_seq.reshape(-1, s_feat)
    scaled_data = scaler.transform(flat_data).reshape(s_batch, s_len, s_feat)
    x_tensor = torch.tensor(scaled_data, dtype=torch.float32)

    with torch.no_grad():
        lstm_out = lstm_model(x_tensor)
        risk_lstm = lstm_out.item()

        reconstructed = ae_model(x_tensor)
        recon_error = ((x_tensor - reconstructed) ** 2).mean().item()
        risk_ae = min(1.0, recon_error / 0.1)  # scaled 0-1

   
    energy_boost = 1.0
    try:
        if "vibration_energy" in feature_names:
            idx = feature_names.index("vibration_energy")
            latest_energy = features_seq[0, -1, idx]
            if latest_energy > 8.0:
                energy_boost = 1.5
    except Exception as e:
        print(f"⚠️ Energy boost error: {e}")

    risk_score = 0.6 * risk_lstm + 0.3 * risk_ae
    risk_score *= energy_boost
    risk_score = min(1.0, risk_score)
    return risk_score


def risk_category(risk_score: float):
    if risk_score < 0.35:
        return "Safe"
    elif risk_score < 0.75:
        return "Warning"
    else:
        return "Danger"


if __name__ == "__main__":
    FEATURE_COLS = [
        "vibration_energy",
        "vibration_entropy",
        "acoustic_energy",
        "acoustic_hit_rate",
        "stress_index",
        "soil_type",
        "excavation_depth",
        "base_geotech_risk"
    ]

   
    data = load_all_data(FEATURE_COLS)


    lstm_model = RockfallLSTM(input_size=len(FEATURE_COLS))
    ae_model = RockfallAutoencoder(input_size=len(FEATURE_COLS))

    all_zones = data["zone_id"].unique()

    print("=== Rockfall Risk Prediction for All Zones ===")
    for zone_id in all_zones:
        try:
            features, _ = load_latest_features_zone(data, zone_id, FEATURE_COLS)
            score = predict_risk(features, FEATURE_COLS, lstm_model, ae_model)
            category = risk_category(score)
            print(f"Zone {zone_id} → Risk Score: {score:.2f}, Category: {category}")
        except Exception as e:
            print(f"❌ Error for {zone_id}: {e}")

