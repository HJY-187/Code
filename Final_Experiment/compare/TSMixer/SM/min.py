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

# Basic configuration parameters (retain optimal TSMixer parameters, only remove hard-coded LOOK_BACK)
PRED_LEN = 12  # Prediction steps, unified with all models
BATCH_SIZE = 64  # Original TSMixer batch size
EPOCHS = 160  # Original TSMixer training epochs
LR = 5e-4  # TSMixer-specific learning rate (should not be too large)
DROPOUT = 0.3  # High Dropout for MLP to prevent overfitting, original parameters retained


# =============================== Dataset Definition (Retain TSMixer core: use all 9-dimensional features) ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        # Retain TSMixer core feature: use all features (9 dimensions) to extract multivariate correlations, different from the first 6 dimensions of other models
        x = self.data[idx:idx + self.seq_len, :]
        # Prediction target: future pred_len steps of the close column (index 3), unified with all models
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, 3]
        # Close price at the last current time step, used for directional loss calculation
        current_price = self.data[idx + self.seq_len - 1, 3]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(current_price, dtype=torch.float32),
        )


# =============================== Data Preprocessing (Retain original logic, complete TR calculation consistent with all models) ===============================
def preprocess_and_split_data(df: pd.DataFrame, look_back: int, pred_len: int):
    df_work = df.copy()
    # Log transformation, unified with all models
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # Complete TR calculation logic consistent with all models to avoid minor data differences
    df_work['TR'] = np.maximum(
        df_work['high'] - df_work['low'],
        np.maximum(
            np.abs(df_work['high'] - df_work['close'].shift(1)),
            np.abs(df_work['low'] - df_work['close'].shift(1))
        )
    )
    # Feature engineering (original TSMixer logic unchanged, unified with all models)
    df_work['ATR'] = df_work['TR'].rolling(window=14).mean()
    delta = df_work['close'] - df_work['close'].shift(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().replace(0, 1e-8)
    rs = avg_gain / avg_loss
    df_work['RSI'] = 100 - (100 / (1 + rs))
    df_work['roll_return'] = (df_work['close'] - df_work['close'].shift(5)) / df_work['close'].shift(5)

    # Remove rolling window null values, unified with all models
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)
    # Extract all 9-dimensional features (TSMixer core feature, retained)
    features = df_work[["open", "high", "low", "close", "volume", "open_interest", "ATR", "RSI", "roll_return"]].values
    dates_work = df_work["datetime"].values  # Date column, unified with all models

    # Training/test set split (8:2+gap, completely unified with all models)
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back  # Avoid overlap, gap is associated with look_back
    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    # Standardization (RobustScaler, unified with all models)
    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    # Get number of features (required by TSMixer, retain return value)
    num_features = features.shape[1]

    # Create data loaders (num_workers=0 for Windows compatibility, unified with all models)
    train_ds = TimeSeriesDataset(train_data, seq_len=look_back, pred_len=pred_len)
    test_ds = TimeSeriesDataset(test_data, seq_len=look_back, pred_len=pred_len)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Return additional num_features to adapt to TSMixer model initialization, other models can ignore it
    return train_loader, test_loader, scaler, dates_work, train_size, look_back, num_features


# =============================== Model Definition: TSMixer (Core structure fully retained, no modifications) ===============================
class TSMixerBlock(nn.Module):
    """TSMixer core module: Time-Mixing + Feature-Mixing, retain all original logic"""
    def __init__(self, seq_len, n_vars, dropout, ff_dim=None):
        super().__init__()
        if ff_dim is None:
            ff_dim = n_vars * 2  # MLP hidden layer width, original parameters retained

        # Time Mixing: [Batch, N_Vars, Seq_Len]
        self.norm_time = nn.BatchNorm1d(n_vars)
        self.lin_time = nn.Linear(seq_len, seq_len)
        self.dropout_time = nn.Dropout(dropout)

        # Feature Mixing: [Batch, Seq_Len, N_Vars]
        self.norm_feat = nn.BatchNorm1d(seq_len)
        self.lin_feat = nn.Sequential(
            nn.Linear(n_vars, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, n_vars),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, N_Vars]
        # Time Mixing + Residual
        res = x.clone()
        x = x.transpose(1, 2)
        x = self.norm_time(x)
        x = self.lin_time(x)
        x = F.relu(x)
        x = self.dropout_time(x)
        x = x.transpose(1, 2)
        x = x + res

        # Feature Mixing + Residual
        res = x.clone()
        x = self.norm_feat(x)
        x = self.lin_feat(x)
        x = x + res
        return x


class TSMixerModel(nn.Module):
    """TSMixer main model, retain all original structures and parameters"""
    def __init__(self, input_size, seq_len, pred_len, n_block=2, dropout=0.1):
        super(TSMixerModel, self).__init__()
        # Stack 2 Mixer Blocks to keep the model lightweight (original parameters retained)
        self.blocks = nn.Sequential(*[
            TSMixerBlock(seq_len=seq_len, n_vars=input_size, dropout=dropout)
            for _ in range(n_block)
        ])
        # Prediction head: Flatten->MLP->prediction steps, original structure retained
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        x = self.blocks(x)
        out = self.head(x)
        return out


# =============================== Loss Function (Completely unified with all models, retain alpha=1.0) ===============================
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


# =============================== Utility Functions (Retain TSMixer-specific logic, unified format) ===============================
def denormalize_predictions(scaled_preds, scaler):
    """Denormalization, retain TSMixer's dynamic feature count acquisition via scaler.n_features_in_ to avoid hardcoding"""
    n, p = scaled_preds.shape
    original = np.zeros_like(scaled_preds)
    num_features = scaler.n_features_in_  # Dynamically get feature count for better adaptability
    for i in range(p):
        dummy = np.zeros((n, num_features))
        dummy[:, 3] = scaled_preds[:, i]  # Close column index 3, unified with all models
        den = scaler.inverse_transform(dummy)
        original[:, i] = np.exp(den[:, 3]) - 1e-8  # Inverse log transformation, unified with all models
    return original


def train_model(model, train_loader, val_loader, epochs=160, lr=5e-4, model_dir='models_tsmixer'):
    """Training function, retain TSMixer-specific AdamW, unified early stopping/printing logic"""
    # Retain TSMixer core: AdamW optimizer (more suitable for MLP structure)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8, verbose=False)
    crit = DirectionalLoss(alpha=1.0)  # Directional loss unified with all models

    best_val_loss = float('inf')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f"{model_dir}/tsmixer_best.pth"
    patience = 0
    early_stop_patience = 15  # Retain original TSMixer early stopping patience value

    for ep in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for bx, by, last_c in train_loader:
            bx, by, last_c = bx.to(device), by.to(device), last_c.to(device)
            opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, by, last_c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping, unified with all models
            opt.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, last_c in val_loader:
                bx, by, last_c = bx.to(device), by.to(device), last_c.to(device)
                pred = model(bx)
                val_loss += crit(pred, by, last_c).item()

        # Calculate average loss and update learning rate
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        sch.step(val_loss)

        # Print every 10 epochs, retain original TSMixer LR printing with unified format
        if (ep + 1) % 10 == 0:
            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch {ep + 1:03d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {ep + 1} (val loss no improvement for {early_stop_patience} epochs)")
                break

    # Load optimal model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def generate_and_save_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path):
    """Generate predictions, unified metric printing format, strictly retain original save path"""
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx = bx.to(device)
            preds_scaled.append(model(bx).cpu().numpy())
            trues_scaled.append(by.numpy())

    # Merge results and denormalize
    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues_scaled = np.concatenate(trues_scaled, axis=0)
    preds = denormalize_predictions(preds_scaled, scaler)
    trues = denormalize_predictions(trues_scaled, scaler)

    # Prediction start index, completely unified with all models (ensure consistent start time)
    start_idx = train_size + 2 * look_back
    valid_len = len(preds)
    res_dates = dates_work[start_idx: start_idx + valid_len]

    # Unified metric printing format, consistent with separators/titles of all models
    print("\n" + "=" * 80)
    print("Evaluation Metrics for Each Prediction Step (Step 1 ~ Step 12)")
    print("=" * 80)
    for step in range(PRED_LEN):
        mse = mean_squared_error(trues[:, step], preds[:, step])
        mae = mean_absolute_error(trues[:, step], preds[:, step])
        rmse = math.sqrt(mse)
        r2 = r2_score(trues[:, step], preds[:, step])
        print(f"Step {step + 1:02d} - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

    # Overall metrics with unified printing format
    print("\n" + "=" * 80)
    print("Overall Prediction Evaluation Metrics (All Steps Combined)")
    print("=" * 80)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(trues, preds)
    print(f"TSMixer Model - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

    # Build output DataFrame, retain original TSMixer column name pred_avg_12steps and add datetime column
    df_out = pd.DataFrame()
    df_out['datetime'] = res_dates  # Time column unified with all models
    for k in range(preds.shape[1]):
        df_out[f'pred_step_{k + 1}'] = preds[:, k]  # Step column names unified with all models
    df_out['pred_avg_12steps'] = preds.mean(axis=1)  # Retain original TSMixer average column name

    # Save CSV: strictly retain original path, specify utf-8 encoding, unified with all models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\nPrediction results saved to: {save_path}")


# =============================== Main Function (Parameters + logic completely unified with all models, command line support) ===============================
def main(data_file="SM_5.csv", output_dir="results_tsmixer", model_dir="models_tsmixer",
         pretrained_path: str | None = None, look_back_override: int | None = None):
    """
    Main function: parameters and operation logic completely unified with DLinear/BiLSTM/fusion models
    Parameters:
    - data_file: Input data file path (CSV)
    - output_dir: Result output directory (for compatibility)
    - model_dir: Model save directory
    - pretrained_path: Pretrained model path (optional)
    - look_back_override: Input sequence length (override default value)
    """
    # Print basic information with unified style
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

    # Process datetime column, delete invalid values (unified with all models)
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    if df["datetime"].isnull().any():
        print("Warning: Some datetime values are invalid and have been removed")
        df = df.dropna(subset=["datetime"])

    # 2. Core: unified dynamic look_back value logic (completely consistent with all models)
    lb_def = 48 if "30" in os.path.basename(data_file) else 20
    look_back = look_back_override if look_back_override is not None else lb_def
    print(f"Input sequence length (look_back): {look_back}")

    # 3. Data preprocessing with exception handling
    print("Starting data preprocessing...")
    try:
        train_loader, test_loader, scaler, dates_work, train_size, look_back, num_features = preprocess_and_split_data(
            df, look_back, PRED_LEN
        )
        print(f"Data preprocessing completed - Training batches: {len(train_loader)}, Test batches: {len(test_loader)}, Number of features: {num_features}")
    except Exception as e:
        print(f"Data preprocessing failed: {e}")
        return

    # 4. Initialize TSMixer model, print total parameters (unified with all models)
    print("Initializing TSMixer Model...")
    model = TSMixerModel(
        input_size=num_features,  # 9-dimensional features, TSMixer core feature
        seq_len=look_back,
        pred_len=PRED_LEN,
        n_block=2,
        dropout=DROPOUT
    ).to(device)
    # Print total model parameters for easy comparison of model complexity
    print(f"Model initialization completed, total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained weights (optional, unified logic with all models)
    if pretrained_path is not None and os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
            print(f"Successfully loaded pretrained weights: {pretrained_path}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")

    # 5. Train model with exception handling
    print("Starting TSMixer model training (AdamW + Time/Feature-Mixing)...")
    try:
        model = train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LR, model_dir=model_dir)
    except Exception as e:
        print(f"Model training failed: {e}")
        return

    # 6. Generate and save predictions [strictly retain original save path]
    print("Generating prediction results...")
    save_path = "results/pred_tsmixer.csv"  # Fully retain original path without any modifications
    try:
        generate_and_save_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path)
    except Exception as e:
        print(f"Prediction generation failed: {e}")
        return

    print("\nAll processes completed!")


# =============================== Execution Entry (Command line parameter support completely unified with all models) ===============================
if __name__ == "__main__":
    # Command line parameter rules are completely consistent with all models:
    # python this_file.py [data_file] [output_dir] [model_dir] [pretrained_path] [look_back]
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "results_tsmixer"
        model_dir = sys.argv[3] if len(sys.argv) > 3 else "models_tsmixer"
        pretrained = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "None" else None
        look_back_arg = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in ("", "None") else None
        main(data_file, output_dir, model_dir, pretrained_path=pretrained, look_back_override=look_back_arg)
    else:
        main()
"""
SM
================================================================================
Step 1 ~ Step 12
================================================================================
Step 01 - MSE: 280.389099, MAE: 11.448013, RMSE: 16.744823, R2: 0.983740
Step 02 - MSE: 453.636658, MAE: 14.369020, RMSE: 21.298748, R2: 0.973700
Step 03 - MSE: 648.893921, MAE: 17.254787, RMSE: 25.473396, R2: 0.962399
Step 04 - MSE: 794.241577, MAE: 18.619055, RMSE: 28.182292, R2: 0.953998
Step 05 - MSE: 940.952637, MAE: 20.277847, RMSE: 30.674951, R2: 0.945519
Step 06 - MSE: 1090.929443, MAE: 21.942440, RMSE: 33.029221, R2: 0.936860
Step 07 - MSE: 1279.223145, MAE: 24.355928, RMSE: 35.766229, R2: 0.925992
Step 08 - MSE: 1418.304443, MAE: 25.640713, RMSE: 37.660383, R2: 0.917970
Step 09 - MSE: 1575.603638, MAE: 27.189917, RMSE: 39.693874, R2: 0.908895
Step 10 - MSE: 1689.222900, MAE: 28.043756, RMSE: 41.100157, R2: 0.902326
Step 11 - MSE: 1819.283691, MAE: 29.253204, RMSE: 42.653062, R2: 0.894794
Step 12 - MSE: 1973.217651, MAE: 30.562456, RMSE: 44.420915, R2: 0.885889
================================================================================
TSMixer Model - MSE: 1163.658569, MAE: 22.413095, RMSE: 34.112440, R2: 0.932673

"""