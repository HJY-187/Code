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

# Fix random seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LEN = 12  # 12-step prediction


# =============================== Dataset Definition (Keep Fully Consistent) ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        x = self.data[idx:idx + self.seq_len, :6]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, 3]
        current_price = self.data[idx + self.seq_len - 1, 3]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(current_price, dtype=torch.float32),
        )


# =============================== Data Preprocessing (Keep Fully Consistent) ===============================
def preprocess_and_split_data(df: pd.DataFrame, look_back: int, pred_len: int):
    df_work = df.copy()
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

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

    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)
    features = df_work[["open", "high", "low", "close", "volume", "open_interest", "ATR", "RSI", "roll_return"]].values
    dates_work = df_work["datetime"].values

    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back
    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    train_ds = TimeSeriesDataset(train_data, seq_len=look_back, pred_len=pred_len)
    test_ds = TimeSeriesDataset(test_data, seq_len=look_back, pred_len=pred_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, test_loader, scaler, dates_work, train_size, look_back


# =============================== PatchTST Core Model ===============================

class PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, patch_len=12, stride=6, d_model=128, nhead=4, num_layers=3,
                 dropout=0.1):
        """
        Args:
            input_dim: Number of input features (6 in this case)
            seq_len: Lookback window length
            pred_len: Prediction length
            patch_len: Length of each patch
            stride: Stride between patches (equal to patch_len for non-overlapping)
            d_model: Internal dimension of Transformer
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride

        # Calculate number of patches N = (L - P) / S + 1
        self.num_patches = int((seq_len - patch_len) / stride) + 1
        # Simple validation to prevent padding issues (assuming seq_len is long enough)
        if self.num_patches <= 0:
            raise ValueError(f"seq_len ({seq_len}) must be greater than patch_len ({patch_len})")

        # 1. Patch Embedding: map patches to d_model dimension
        # Input shape: [Batch * input_dim, num_patches, patch_len]
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # 2. Positional Embedding: learnable positional encoding
        self.position_embedding = nn.Parameter(torch.randn(1, input_dim, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder Backbone
        # batch_first=True means input shape is [Batch, Seq, Feature]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Flatten Head: flatten Transformer output and project to prediction length
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, L, C = x.shape

        # --- A. RevIN (Reversible Instance Normalization) ---
        # Address distribution shift, a standard component of PatchTST
        # Calculate mean and standard deviation (per instance and channel)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # --- B. Patching & Channel Independence ---
        # Reshape to [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1)

        # Unfold operation to generate patches: [Batch, Channels, Num_Patches, Patch_Len]
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # Merge Batch and Channel dimensions to achieve Channel Independence (weight sharing)
        # Reshaped to: [Batch * Channels, Num_Patches, Patch_Len]
        x = x.reshape(B * C, self.num_patches, self.patch_len)

        # --- C. Embedding & Transformer ---
        # Linear Projection
        enc_out = self.patch_embedding(x)  # [B*C, N, d_model]

        # Add Positional Embedding
        # pos_emb: [1, C, N, d_model] -> expand -> [B, C, N, d_model] -> reshape [B*C, N, d_model]
        pos_emb = self.position_embedding.repeat(B, 1, 1, 1).reshape(B * C, self.num_patches, -1)
        enc_out = self.dropout(enc_out + pos_emb)

        # Transformer Encoder
        enc_out = self.transformer_encoder(enc_out)  # [B*C, N, d_model]

        # --- D. Prediction Head ---
        # Flatten: [B*C, N * d_model]
        enc_out = enc_out.reshape(B * C, -1)
        # Linear Projection: [B*C, pred_len]
        dec_out = self.head(enc_out)

        # Reshape back to [Batch, Channels, Pred_Len]
        dec_out = dec_out.reshape(B, C, self.pred_len)

        # --- E. RevIN Denormalization ---
        # Denormalization: x * std + mean
        # Note broadcasting mechanism: dec_out [B, C, P], stdev [B, 1, C] -> permute to match
        stdev = stdev.permute(0, 2, 1)  # [B, C, 1]
        means = means.permute(0, 2, 1)  # [B, C, 1]

        dec_out = dec_out * stdev + means

        # --- F. Select Target Channel ---
        # Our Dataset labels only include close price (usually 4th column, index 3)
        # PatchTST computes predictions for all channels, we only take the one we need
        # Input feature order: open, high, low, close, volume, oi (total 6)
        target_channel = 3
        output = dec_out[:, target_channel, :]  # [Batch, Pred_Len]

        return output


# =============================== Loss Function (Keep Fully Consistent) ===============================
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


# =============================== Utility Functions (Keep Fully Consistent) ===============================
def denormalize_predictions(scaled_preds, scaler):
    n, p = scaled_preds.shape
    original = np.zeros_like(scaled_preds)
    for i in range(p):
        dummy = np.zeros((n, 9))
        dummy[:, 3] = scaled_preds[:, i]
        den = scaler.inverse_transform(dummy)
        original[:, i] = np.exp(den[:, 3]) - 1e-8
    return original


def train_model(model, train_loader, val_loader, epochs=160, lr=1e-3, model_dir='models_patchtst'):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8, verbose=False)
    crit = DirectionalLoss(alpha=1.0)
    best_val_loss = float('inf')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f"{model_dir}/patchtst_best.pth"
    patience = 0
    early_stop_patience = 12

    print("Starting PatchTST model training...")
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by, last_close in train_loader:
            bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
            opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, by, last_close)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, last_close in val_loader:
                bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
                pred = model(bx)
                val_loss += crit(pred, by, last_close).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        sch.step(val_loss)

        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep + 1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stop at epoch {ep + 1} (val loss no improvement)")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def generate_and_save_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path):
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx = bx.to(device)
            preds_scaled.append(model(bx).cpu().numpy())
            trues_scaled.append(by.numpy())

    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues_scaled = np.concatenate(trues_scaled, axis=0)
    preds = denormalize_predictions(preds_scaled, scaler)
    trues = denormalize_predictions(trues_scaled, scaler)

    start_idx = train_size + 2 * look_back
    valid_len = len(preds)
    res_dates = dates_work[start_idx: start_idx + valid_len]

    print("\n" + "=" * 80)
    print("Evaluation Metrics for Each Prediction Step (Step 1 ~ Step 12)")
    print("=" * 80)
    for step in range(PRED_LEN):
        mse_step = mean_squared_error(trues[:, step], preds[:, step])
        mae_step = mean_absolute_error(trues[:, step], preds[:, step])
        rmse_step = math.sqrt(mse_step)
        r2_step = r2_score(trues[:, step], preds[:, step])
        print(
            f"Step {step + 1:02d} - MSE: {mse_step:.6f}, MAE: {mae_step:.6f}, RMSE: {rmse_step:.6f}, R2: {r2_step:.6f}")

    print("\n" + "=" * 80)
    print("Overall Prediction Evaluation Metrics (All Steps Combined)")
    print("=" * 80)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(trues, preds)
    print(f"PatchTST Model - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

    df_out = pd.DataFrame()
    df_out['datetime'] = res_dates
    for k in range(preds.shape[1]):
        df_out[f'pred_step_{k + 1}'] = preds[:, k]
    df_out['pred_avg_1hr'] = preds.mean(axis=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\nPrediction results saved to: {save_path}")


# =============================== Main Function ===============================
def main(data_file="I_5.csv", output_dir="results_patchtst", model_dir="models_patchtst",
         pretrained_path: str | None = None, look_back_override: int | None = None):
    print(f"Current device used: {device}")
    print(f"Input data file: {data_file}")

    try:
        df = pd.read_csv(data_file)
        print(f"Successfully loaded data with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found")
        return
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    if df["datetime"].isnull().any():
        print("Warning: Some datetime values are invalid and have been removed")
        df = df.dropna(subset=["datetime"])

    # Default Look back
    lb_def = 48 if "30" in os.path.basename(data_file) else 20
    look_back = look_back_override if look_back_override is not None else lb_def
    print(f"Input sequence length (look_back): {look_back}")

    print("Starting data preprocessing...")
    try:
        train_loader, test_loader, scaler, dates_work, train_size, look_back = preprocess_and_split_data(
            df, look_back, PRED_LEN
        )
    except Exception as e:
        print(f"Failed to preprocess data: {e}")
        return

    # Model initialization: PatchTST
    print("Initializing PatchTST Model...")
    # Parameter settings:
    # patch_len=12, stride=6.
    # If look_back=20, num_patches = (20-12)/6 + 1 = 2 (patches)
    # If look_back=48, num_patches = (48-12)/6 + 1 = 7 (patches)
    model = PatchTST(
        input_dim=6,
        seq_len=look_back,
        pred_len=PRED_LEN,
        patch_len=12,
        stride=6,
        d_model=64,  # Hidden layer dimension
        nhead=4,
        num_layers=3,
        dropout=0.3
    ).to(device)

    print(f"Model initialization completed, total parameters: {sum(p.numel() for p in model.parameters()):,}")

    if pretrained_path is not None and os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
            print(f"Successfully loaded pretrained weights: {pretrained_path}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")

    try:
        model = train_model(model, train_loader, test_loader, epochs=160, lr=1e-3, model_dir=model_dir)
    except Exception as e:
        print(f"Failed to train model: {e}")
        return

    print("Generating prediction results...")
    save_path = "results/pred_patchtst.csv"
    try:
        generate_and_save_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path)
    except Exception as e:
        print(f"Failed to generate prediction results: {e}")
        return

    print("\nAll processes completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "results_patchtst"
        model_dir = sys.argv[3] if len(sys.argv) > 3 else "models_patchtst"
        pretrained = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "None" else None
        look_back_arg = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in ("", "None") else None
        main(data_file, output_dir, model_dir, pretrained_path=pretrained, look_back_override=look_back_arg)
    else:
        main()
"""
I
================================================================================
Step 1 ~ Step 12
================================================================================
Step 01 - MSE: 7.298864, MAE: 1.845311, RMSE: 2.701641, R2: 0.995371
Step 02 - MSE: 10.926790, MAE: 2.253954, RMSE: 3.305570, R2: 0.993068
Step 03 - MSE: 14.477614, MAE: 2.604762, RMSE: 3.804946, R2: 0.990814
Step 04 - MSE: 17.956060, MAE: 2.907156, RMSE: 4.237459, R2: 0.988604
Step 05 - MSE: 21.405943, MAE: 3.186205, RMSE: 4.626656, R2: 0.986412
Step 06 - MSE: 24.894194, MAE: 3.447478, RMSE: 4.989408, R2: 0.984195
Step 07 - MSE: 28.420170, MAE: 3.700587, RMSE: 5.331057, R2: 0.981954
Step 08 - MSE: 31.858246, MAE: 3.945170, RMSE: 5.644311, R2: 0.979767
Step 09 - MSE: 35.285198, MAE: 4.181024, RMSE: 5.940135, R2: 0.977586
Step 10 - MSE: 38.578323, MAE: 4.394784, RMSE: 6.211145, R2: 0.975490
Step 11 - MSE: 41.832287, MAE: 4.596334, RMSE: 6.467788, R2: 0.973418
Step 12 - MSE: 45.132313, MAE: 4.790230, RMSE: 6.718059, R2: 0.971314

PatchTST Model - MSE: 26.505487, MAE: 3.487749, RMSE: 5.148348, R2: 0.983166

"""