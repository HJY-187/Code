import os
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================== Configuration Parameters ===============================
LOOK_BACK = 22
PRED_LEN = 1
EPOCHS = 180
LR = 1e-3
BATCH_SIZE = 64
DATA_FILE = "JM_daily.csv"
OUTPUT_FILE = "pred_daily_patchtst.csv"


# =============================== Dataset ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - PRED_LEN + 1

    def __getitem__(self, idx: int):
        # Input: All features of the previous seq_len rows
        x = self.data[idx: idx + self.seq_len, :]
        # Target: Close value at seq_len position (index 3)
        y = self.data[idx + self.seq_len, 3]
        # Closing price at the current moment
        previous_close = self.data[idx + self.seq_len - 1, 3]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
            torch.tensor(previous_close, dtype=torch.float32)
        )


# =============================== Data Preprocessing ===============================
def preprocess_data(df: pd.DataFrame, look_back: int):
    df_work = df.copy()

    # 1. Log Transformation
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # 2. Add Calendar Features
    dt = df_work['datetime']
    df_work['day_of_week'] = dt.dt.dayofweek
    df_work['day_of_year'] = dt.dt.dayofyear
    df_work['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df_work['month'] = dt.dt.month

    # 3. Technical Indicator Calculation
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

    # 4. Data Cleaning
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # 5. Extract Features and Timestamps
    feature_cols = [
        "open", "high", "low", "close", "volume", "open_interest",
        "day_of_week", "day_of_year", "week_of_year", "month",
        "ATR", "RSI", "roll_return"
    ]
    features = df_work[feature_cols].values

    # 6. Split Training/Test Sets
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    test_start_idx = train_size + gap
    num_test_samples = len(raw_test) - look_back - PRED_LEN + 1
    target_indices = [test_start_idx + i + look_back for i in range(num_test_samples)]
    test_target_dates = df_work['datetime'].iloc[target_indices].values

    # 7. Normalization
    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    # 8. DataLoader
    train_ds = TimeSeriesDataset(train_data, seq_len=look_back)
    test_ds = TimeSeriesDataset(test_data, seq_len=look_back)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler, test_target_dates, len(feature_cols)


# =============================== Model: PatchTST (Fixed Version) ===============================
class PatchTST(nn.Module):
    def __init__(self, in_features, seq_len, pred_len, patch_len=8, stride=4, d_model=64, nhead=4, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_features = in_features
        self.patch_len = patch_len
        self.stride = stride
        self.target_col_index = 3  # Close index

        # Calculate Number of Patches
        self.num_patches = int((seq_len - patch_len) / stride) + 1

        # 1. Patch Embedding
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # 2. Positional Embedding (Fix: Changed to 3D tensor [1, Num_Patches, d_model])
        # This way it will be correctly broadcast to [Batch*Channels, Num_Patches, d_model]
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        self.dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Flatten Head
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        B, L, C = x.shape

        # --- RevIN (Normalization) ---
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # --- Patching & Channel Independence ---
        # Reshape to [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1)

        # Unfold to generate Patches: [Batch, Channels, Num_Patches, Patch_Len]
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # Fuse Batch and Channel: [Batch * Channels, Num_Patches, Patch_Len]
        x = x.reshape(B * C, self.num_patches, self.patch_len)

        # --- Transformer ---
        enc_out = self.patch_embedding(x)  # [B*C, N, d_model]

        # Add positional encoding (now dimensions are correct, kept as 3D tensor)
        enc_out = enc_out + self.position_embedding
        enc_out = self.dropout(enc_out)

        # Transformer input must be 3D: [Batch, Seq, Dim]
        enc_out = self.transformer_encoder(enc_out)  # [B*C, N, d_model]

        # --- Head ---
        enc_out = enc_out.reshape(B * C, -1)  # Flatten
        dec_out = self.head(enc_out)  # [B*C, Pred_Len]

        # Restore shape [Batch, Channels, Pred_Len]
        dec_out = dec_out.reshape(B, C, self.pred_len)

        # --- RevIN Denormalization ---
        stdev = stdev.permute(0, 2, 1)  # [B, C, 1]
        means = means.permute(0, 2, 1)  # [B, C, 1]
        dec_out = dec_out * stdev + means

        # --- Select Target Column ---
        output = dec_out[:, self.target_col_index, :]  # [Batch, Pred_Len]

        return output


class DirectionalLoss(nn.Module):
    def __init__(self, alpha=2.0):
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


# =============================== Training and Utility Functions ===============================
def denormalize_close(scaled_val, scaler, num_features):
    dummy = np.zeros((1, num_features))
    dummy[0, 3] = scaled_val  # Close at index 3
    denorm = scaler.inverse_transform(dummy)[0, 3]
    return np.exp(denorm) - 1e-8


def run_process():
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 1. Prepare Data
    train_loader, test_loader, scaler, test_target_dates, num_features = preprocess_data(df, LOOK_BACK)
    print(f"Data prepared. Using {num_features} features. Test samples: {len(test_target_dates)}")

    # 2. Initialize Model
    model = PatchTST(
        in_features=num_features,
        seq_len=LOOK_BACK,
        pred_len=PRED_LEN,
        patch_len=8,
        stride=4,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.2
    ).to(device)

    print(f"Model initialized: PatchTST. Params: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Training Configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    criterion = DirectionalLoss(alpha=2)

    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by, last_close in train_loader:
            bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by, last_close)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, last_close in test_loader:
                bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by, last_close).item()
        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 4. Evaluation and Saving
    print("Training finished. Evaluating best model...")
    model.load_state_dict(best_model_weights)
    model.eval()

    preds_scaled, trues_scaled, prev_closes_scaled = [], [], []

    with torch.no_grad():
        for bx, by, b_prev_c in test_loader:
            bx = bx.to(device)
            output = model(bx)
            preds_scaled.extend(output.cpu().numpy().flatten())
            trues_scaled.extend(by.numpy().flatten())
            prev_closes_scaled.extend(b_prev_c.numpy().flatten())

    final_preds, final_trues, final_prev_closes = [], [], []
    loop_len = min(len(preds_scaled), len(trues_scaled), len(prev_closes_scaled))
    for i in range(loop_len):
        final_preds.append(denormalize_close(preds_scaled[i], scaler, num_features))
        final_trues.append(denormalize_close(trues_scaled[i], scaler, num_features))
        final_prev_closes.append(denormalize_close(prev_closes_scaled[i], scaler, num_features))

    final_trues_np = np.array(final_trues)
    final_preds_np = np.array(final_preds)
    final_prev_closes_np = np.array(final_prev_closes)

    mae = mean_absolute_error(final_trues_np, final_preds_np)
    mse = mean_squared_error(final_trues_np, final_preds_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(final_trues_np, final_preds_np)

    true_trend = np.sign(final_trues_np - final_prev_closes_np)
    pred_trend = np.sign(final_preds_np - final_prev_closes_np)
    trend_acc = np.mean(true_trend == pred_trend) * 100

    print("\n" + "=" * 40)
    print(" Evaluation Metrics (Test Set) ")
    print("=" * 40)
    print(f" MAE          : {mae:.4f}")
    print(f" RMSE         : {rmse:.4f}")
    print(f" MSE          : {mse:.4f}")
    print(f" R^2          : {r2:.4f}")
    print(f" Trend Acc    : {trend_acc:.2f}%")
    print("=" * 40 + "\n")

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    min_len = min(len(test_target_dates), len(final_preds))
    results_df = pd.DataFrame({
        "datetime": test_target_dates[:min_len],
        "pred_close": final_preds[:min_len]
    })
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved predictions to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_process()

"""
JM
========================================
 Evaluation Metrics (Test Set) 
========================================
 MAE          : 36.6250
 RMSE         : 47.1894
 MSE          : 2226.8379
 R^2          : 0.9629
 Trend Acc    : 52.60%
========================================
"""