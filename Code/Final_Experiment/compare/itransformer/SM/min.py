import os
import math
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fix random seeds (consistent with all models)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LEN = 12  # 12-step prediction, corresponding to 1 hour, consistent with all models


# =============================== Dataset Definition (completely unified with all models, using first 6 dimensions of features) ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        # Keep original iTransformer logic: use first 6 dimensions of features, different from TSMixer's 9 dimensions
        x = self.data[idx:idx + self.seq_len, :6]
        # Prediction target: future PRED_LEN steps of close column (index 3), consistent with all models
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, 3]
        # Close price at the last time step, used for directional loss calculation
        current_price = self.data[idx + self.seq_len - 1, 3]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(current_price, dtype=torch.float32),
        )


# =============================== Data Preprocessing (incorporate feature engineering, completely unified with all models) ===============================
def preprocess_and_split_data(df: pd.DataFrame, look_back: int, pred_len: int):
    df_work = df.copy()
    # Log transformation to avoid price scale issues, consistent with all models
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # Complete feature engineering (merge originally scattered logic, completely consistent calculation with all models)
    df_work['TR'] = np.maximum(
        df_work['high'] - df_work['low'],
        np.maximum(
            np.abs(df_work['high'] - df_work['close'].shift(1)),
            np.abs(df_work['low'] - df_work['close'].shift(1))
        )
    )
    df_work['ATR'] = df_work['TR'].rolling(window=14).mean()
    delta = df_work['close'] - df_work['close'].shift(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().replace(0, 1e-8)
    rs = avg_gain / avg_loss
    df_work['RSI'] = 100 - (100 / (1 + rs))
    df_work['roll_return'] = (df_work['close'] - df_work['close'].shift(5)) / df_work['close'].shift(5)

    # Remove rolling window null values and reset index, consistent with all models
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)
    # Extract 9-dimensional features (for standardization, consistent with all models)
    features = df_work[["open", "high", "low", "close", "volume", "open_interest", "ATR", "RSI", "roll_return"]].values
    dates_work = df_work["datetime"].values  # Date column for subsequent prediction time alignment

    # Train/test set split (8:2+gap to avoid data overlap, completely unified with all models)
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back
    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    # Standardization (RobustScaler is robust to outliers, consistent with all models)
    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    # Create data loaders (num_workers=0 for Windows compatibility, consistent with all models)
    train_ds = TimeSeriesDataset(train_data, seq_len=look_back, pred_len=pred_len)
    test_ds = TimeSeriesDataset(test_data, seq_len=look_back, pred_len=pred_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, test_loader, scaler, dates_work, train_size, look_back


# =============================== Model Definition: iTransformer (Core structure fully retained, no modifications) ===============================
class PositionalEncoding(nn.Module):
    """Positional encoding layer, retain original iTransformer logic"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class iTransformerModel(nn.Module):
    """iTransformer core model, retain all original features: features as tokens, reversed dimensions, norm_first=True"""
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=2, pred_len=12, dropout=0.1, seq_len=48):
        super(iTransformerModel, self).__init__()
        self.seq_len = seq_len

        # iTransformer core: encode complete history of each feature as a Token
        self.feature_embedding = nn.Linear(seq_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Retain original optimization configuration: norm_first=True, dim_feedforward=d_model*4
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # iTransformer exclusive optimization, faster convergence
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Output head is completely consistent with original model, mapped to prediction steps
        self.decoder = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # Input format: [Batch, Seq_Len, Features], compatible with all models
        B, L, D = x.shape

        # iTransformer core modification: reverse time and feature dimensions (time->feature, feature->time)
        x = x.transpose(1, 2)  # -> [Batch, Features, Seq_Len]
        # Map complete history of each feature to vector
        x = self.feature_embedding(x)  # -> [Batch, Features, d_model]

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Fuse representations of all features with average pooling
        x = x.mean(dim=1)  # -> [Batch, d_model]
        return self.decoder(x)


# =============================== Loss Function (Unified directional loss with all models) ===============================
class DirectionalLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.base_loss = nn.SmoothL1Loss()

    def forward(self, pred, true, last_close):
        num_loss = self.base_loss(pred, true)
        last_close = last_close.view(-1, 1)
        true_diff = true - last_close
        pred_diff = pred - last_close
        penalty = F.relu(-1.0 * true_diff * pred_diff)
        return num_loss + self.alpha * torch.mean(penalty)


# =============================== Utility Functions (Unified with all models, retain denormalization logic) ===============================
def denormalize_predictions(scaled_preds, scaler):
    """Denormalization to restore original price scale (log + inverse standardization), retain original logic"""
    n, p = scaled_preds.shape
    original = np.zeros_like(scaled_preds)
    for i in range(p):
        dummy = np.zeros((n, 9))  # 9-dimensional features, consistent with all models
        dummy[:, 3] = scaled_preds[:, i]  # close column index 3
        den = scaler.inverse_transform(dummy)
        original[:, i] = np.exp(den[:, 3]) - 1e-8  # Inverse log transformation
    return original


def train_model(model, train_loader, val_loader, epochs=160, lr=5e-4, model_dir='models_itransformer'):
    """Training function: upgraded to early stopping + learning rate scheduling, retain original iTransformer LR/gradient clipping, unified with all models"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8, verbose=False)
    crit = DirectionalLoss(alpha=1.0)  # Unified directional loss with all models
    best_val_loss = float('inf')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f"{model_dir}/itransformer_best.pth"
    patience = 0
    early_stop_patience = 12  # Early stopping patience, consistent with ensemble model/DLinear

    print("Start training iTransformer model (features as tokens + reversed dimensions)...")
    for ep in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for bx, by, last_close in train_loader:
            bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
            opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, by, last_close)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Retain original gradient clipping for stable training
            opt.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, last_close in val_loader:
                bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
                pred = model(bx)
                val_loss += crit(pred, by, last_close).item()

        # Calculate average loss and update learning rate
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        sch.step(val_loss)

        # Print every 5 epochs, retain original print frequency with unified format
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep + 1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping logic to avoid overfitting
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stop at epoch {ep + 1} (val loss no improvement for {early_stop_patience} epochs)")
                break

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def generate_and_save_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path):
    """Generate predictions: fix date index error, unified metric printing, strictly retain original save path"""
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx = bx.to(device)
            preds_scaled.append(model(bx).cpu().numpy())
            trues_scaled.append(by.numpy())

    # Merge prediction results and denormalize to restore original prices
    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues_scaled = np.concatenate(trues_scaled, axis=0)
    preds = denormalize_predictions(preds_scaled, scaler)
    trues = denormalize_predictions(trues_scaled, scaler)

    # Core fix: unified prediction start index, completely consistent with all models (ensure time alignment)
    start_idx = train_size + 2 * look_back
    valid_len = len(preds)
    res_dates = dates_work[start_idx: start_idx + valid_len]

    # Unified metric printing format, consistent with separator/title/step format of all models
    print("\n" + "=" * 80)
    print("Evaluation metrics for each prediction step (Step 1 ~ Step 12)")
    print("=" * 80)
    for step in range(PRED_LEN):
        mse_step = mean_squared_error(trues[:, step], preds[:, step])
        mae_step = mean_absolute_error(trues[:, step], preds[:, step])
        rmse_step = math.sqrt(mse_step)
        r2_step = r2_score(trues[:, step], preds[:, step])
        print(f"Step {step + 1:02d} - MSE: {mse_step:.6f}, MAE: {mae_step:.6f}, RMSE: {rmse_step:.6f}, R2: {r2_step:.6f}")

    # Overall evaluation metrics, unified printing format with all models
    print("\n" + "=" * 80)
    print("Overall prediction evaluation metrics (all steps combined)")
    print("=" * 80)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(trues, preds)
    print(f"iTransformer Model - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

    # Build output DF with completely unified column names (datetime + pred_step_1~12 + pred_avg_1hr)
    df_out = pd.DataFrame()
    df_out['datetime'] = res_dates
    for k in range(preds.shape[1]):
        df_out[f'pred_step_{k + 1}'] = preds[:, k]
    df_out['pred_avg_1hr'] = preds.mean(axis=1)  # 1-hour average prediction value, unified with all models

    # Save CSV: strictly retain original path, specify utf-8 encoding, unified with all models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\nPrediction results saved to: {save_path}")


# =============================== Main Function (Completely unified parameters + logic with all models, support command line) ===============================
def main(data_file="SM_5.csv", output_dir="results_itransformer", model_dir="models_itransformer",
         pretrained_path: str | None = None, look_back_override: int | None = None):
    """
    Main function: completely unified parameters and running logic with ensemble model/DLinear/BiLSTM/TSMixer
    Parameters:
    - data_file: Input data file path (CSV)
    - output_dir: Result output directory (for compatibility)
    - model_dir: Model save directory
    - pretrained_path: Pretrained model path (optional)
    - look_back_override: Input sequence length (override default value)
    """
    # Print basic information with unified style across all models
    print(f"Current device: {device}")
    print(f"Input data file: {data_file}")

    # 1. Load data with full-process exception handling
    try:
        df = pd.read_csv(data_file)
        print(f"Successfully loaded data with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found")
        return
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # Process datetime column and remove invalid values, unified with all models
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    if df["datetime"].isnull().any():
        print("Warning: Some datetime formats are invalid and have been removed")
        df = df.dropna(subset=["datetime"])

    # 2. Core: unified look_back dynamic value selection logic (completely consistent with all models)
    lb_def = 48 if "30" in os.path.basename(data_file) else 20
    look_back = look_back_override if look_back_override is not None else lb_def
    print(f"Input sequence length (look_back): {look_back}")

    # 3. Data preprocessing with exception handling
    print("Start data preprocessing...")
    try:
        train_loader, test_loader, scaler, dates_work, train_size, look_back = preprocess_and_split_data(
            df, look_back, PRED_LEN
        )
        print(f"Data preprocessing completed - Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Data preprocessing failed: {e}")
        return

    # 4. Initialize iTransformer model and print total parameters, unified with all models
    print("Initializing iTransformer Model...")
    model = iTransformerModel(
        input_dim=6, d_model=64, nhead=4, num_layers=2,
        pred_len=PRED_LEN, seq_len=look_back, dropout=0.1
    ).to(device)
    # Print total number of model parameters for comparing complexity across models
    print(f"Model initialization completed, total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained weights (optional, unified logic with all models)
    if pretrained_path is not None and os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
            print(f"Successfully loaded pretrained weights: {pretrained_path}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")

    # 5. Train model with exception handling
    try:
        model = train_model(model, train_loader, test_loader, epochs=160, lr=5e-4, model_dir=model_dir)
    except Exception as e:
        print(f"Model training failed: {e}")
        return

    # 6. Generate and save prediction results [strictly retain original save path]
    print("Start generating prediction results...")
    save_path = "results/pred_itransformer.csv"  # Fully retain original path without any modification
    try:
        generate_and_save_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path)
    except Exception as e:
        print(f"Prediction result generation failed: {e}")
        return

    print("\nAll processes completed!")


# =============================== Running Entry (Completely unified command line argument support with all models) ===============================
if __name__ == "__main__":
    # Command line argument rules are completely consistent with all models:
    # python this_file.py [data_file] [output_dir] [model_dir] [pretrained_path] [look_back]
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "results_itransformer"
        model_dir = sys.argv[3] if len(sys.argv) > 3 else "models_itransformer"
        pretrained = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "None" else None
        look_back_arg = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in ("", "None") else None
        main(data_file, output_dir, model_dir, pretrained_path=pretrained, look_back_override=look_back_arg)
    else:

        main()
"""
SM
Step 1 ~ Step 12
================================================================================
Step 01 - MSE: 456.561432, MAE: 14.622072, RMSE: 21.367298, R2: 0.973523
Step 02 - MSE: 634.914673, MAE: 16.950361, RMSE: 25.197513, R2: 0.963190
Step 03 - MSE: 808.001282, MAE: 19.097300, RMSE: 28.425363, R2: 0.953179
Step 04 - MSE: 993.700134, MAE: 21.333162, RMSE: 31.523010, R2: 0.942445
Step 05 - MSE: 1130.081543, MAE: 22.588007, RMSE: 33.616685, R2: 0.934569
Step 06 - MSE: 1297.736206, MAE: 24.412544, RMSE: 36.024106, R2: 0.924891
Step 07 - MSE: 1450.058838, MAE: 25.888882, RMSE: 38.079638, R2: 0.916108
Step 08 - MSE: 1602.718872, MAE: 27.431463, RMSE: 40.033971, R2: 0.907304
Step 09 - MSE: 1751.600342, MAE: 28.727854, RMSE: 41.852125, R2: 0.898718
Step 10 - MSE: 1899.478271, MAE: 30.150139, RMSE: 43.583004, R2: 0.890169
Step 11 - MSE: 2057.939697, MAE: 31.577881, RMSE: 45.364520, R2: 0.880993
Step 12 - MSE: 2249.248047, MAE: 33.383259, RMSE: 47.426238, R2: 0.869926

================================================================================
iTransformer Model - MSE: 1361.002808, MAE: 24.680239, RMSE: 36.891772, R2: 0.921251
"""