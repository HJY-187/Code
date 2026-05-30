import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Configuration Parameters =================
LOOK_BACK = 22
PRED_LEN = 1
EPOCHS = 200  # Slightly increase the maximum number of epochs to let early stopping take over
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 15  # [New] Early stopping patience: number of Epochs with no improvement before stopping
DATA_FILE = "J_daily.csv"
OUTPUT_FILE = "pred_daily_itransformer.csv"


# ================= MODEL: iTransformer =================
class iTransformerModel(nn.Module):
    def __init__(self, seq_len, in_features, pred_len=1, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # iTransformer core: Embedding for each variate, not for time steps
        self.enc_embedding = nn.Linear(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Vars]
        # iTransformer requires: [Batch, Vars, Seq_Len]
        x = x.permute(0, 2, 1)

        x_enc = self.enc_embedding(x)  # [Batch, Vars, d_model]
        enc_out = self.encoder(x_enc)  # [Batch, Vars, d_model]
        dec_out = self.projector(enc_out)  # [Batch, Vars, Pred_Len]

        dec_out = dec_out.permute(0, 2, 1)  # [Batch, Pred_Len, Vars]

        # We only need to predict Close (assuming Close is the 4th column, index 3)
        return dec_out[:, 0, 3].unsqueeze(1)


# ================= Dataset =================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - PRED_LEN + 1

    def __getitem__(self, idx: int):
        # Input: all features of the previous seq_len rows
        x = self.data[idx: idx + self.seq_len, :]
        # Target: close value at seq_len position (index 3)
        y = self.data[idx + self.seq_len, 3]
        # Previous day's close price
        previous_close = self.data[idx + self.seq_len - 1, 3]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
            torch.tensor(previous_close, dtype=torch.float32)
        )


# ================= Data Preprocessing =================
def preprocess_data(df: pd.DataFrame, look_back: int):
    df_work = df.copy()

    # 1. Log transformation
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # 2. Add calendar features
    dt = df_work['datetime']
    df_work['day_of_week'] = dt.dt.dayofweek
    df_work['day_of_year'] = dt.dt.dayofyear
    df_work['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df_work['month'] = dt.dt.month

    # 3. Technical indicator calculation
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

    # 4. Data cleaning
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # 5. Feature extraction
    feature_cols = [
        "open", "high", "low", "close", "volume", "open_interest",
        "day_of_week", "day_of_year", "week_of_year", "month",
        "ATR", "RSI", "roll_return"
    ]
    features = df_work[feature_cols].values

    # 6. Split training/test sets
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    # Extract target timestamps corresponding to test set
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


def denormalize_close(scaled_val, scaler, num_features):
    dummy = np.zeros((1, num_features))
    dummy[0, 3] = scaled_val  # Close is at index 3
    denorm = scaler.inverse_transform(dummy)[0, 3]
    return np.exp(denorm) - 1e-8


# ================= Main Process =================
def run_process():
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 1. Prepare data
    train_loader, test_loader, scaler, test_target_dates, num_features = preprocess_data(df, LOOK_BACK)
    print(f"Data prepared. Features: {num_features}. Test samples: {len(test_target_dates)}")

    # 2. Initialize iTransformer
    model = iTransformerModel(
        seq_len=LOOK_BACK,
        in_features=num_features,
        d_model=128,
        nhead=4,
        num_layers=2
    ).to(device)

    # 3. Training
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_weights = None

    # [New] Early stopping counter
    early_stop_counter = 0

    print(f"Starting training (Max Epochs: {EPOCHS}, Patience: {PATIENCE})...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, _ in test_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += loss_fn(pred, by).item()
        val_loss /= len(test_loader)

        # Learning rate adjustment
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # [Modified] Save best model and early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0  # Loss improved, reset counter
        else:
            early_stop_counter += 1  # Loss not improved, increment counter
            # Optional: Print early stopping progress
            # print(f"  [EarlyStopping] Counter: {early_stop_counter}/{PATIENCE}")

        # Trigger early stopping
        if early_stop_counter >= PATIENCE:
            print(f"\n>> Early stopping triggered at Epoch {epoch + 1}. Best Val Loss: {best_loss:.6f}")
            break

    # 4. Evaluation
    if best_weights is not None:
        print("Restoring best weights...")
        model.load_state_dict(best_weights)
    else:
        print("Warning: No best weights found (possibly all NaN loss?). Using last weights.")

    model.eval()
    preds_scaled, trues_scaled, prev_closes_scaled = [], [], []

    with torch.no_grad():
        for bx, by, b_prev_c in test_loader:
            bx = bx.to(device)
            pred = model(bx)
            preds_scaled.extend(pred.cpu().numpy().flatten())
            trues_scaled.extend(by.numpy().flatten())
            prev_closes_scaled.extend(b_prev_c.numpy().flatten())

    # 5. Denormalization
    final_preds, final_trues, final_prev_closes = [], [], []
    loop_len = min(len(preds_scaled), len(trues_scaled), len(prev_closes_scaled))
    for i in range(loop_len):
        final_preds.append(denormalize_close(preds_scaled[i], scaler, num_features))
        final_trues.append(denormalize_close(trues_scaled[i], scaler, num_features))
        final_prev_closes.append(denormalize_close(prev_closes_scaled[i], scaler, num_features))

    # 6. Calculate metrics
    final_trues_np = np.array(final_trues)
    final_preds_np = np.array(final_preds)
    final_prev_closes_np = np.array(final_prev_closes)

    mae = mean_absolute_error(final_trues_np, final_preds_np)
    mse = mean_squared_error(final_trues_np, final_preds_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(final_trues_np, final_preds_np)

    # Trend accuracy
    true_trend = np.sign(final_trues_np - final_prev_closes_np)
    pred_trend = np.sign(final_preds_np - final_prev_closes_np)
    trend_acc = np.mean(true_trend == pred_trend) * 100

    print("\n" + "=" * 40)
    print(" iTransformer Daily Results (with Early Stopping)")
    print("=" * 40)
    print(f" MAE      : {mae:.4f}")
    print(f" RMSE     : {rmse:.4f}")
    print(f" R2       : {r2:.4f}")
    print(f" Trend Acc: {trend_acc:.2f}%")
    print("=" * 40 + "\n")

    # 7. Save file
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
J
========================================
 iTransformer Daily Results (with Early Stopping)
========================================
 MAE      : 38.7814
 RMSE     : 49.6367
 R2       : 0.9606
 Trend Acc: 51.02%
========================================
"""