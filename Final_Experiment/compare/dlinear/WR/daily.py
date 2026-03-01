import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================== Configuration Parameters ===============================
LOOK_BACK = 22  # Look-back window
PRED_LEN = 1  # Prediction length
EPOCHS = 180  # Training epochs
LR = 1e-3  # Learning rate (DLinear has simple structure, can use slightly larger learning rate)
BATCH_SIZE = 64
DATA_FILE = "WR_daily.csv"
OUTPUT_FILE = "pred_daily_simple_dlinear.csv"


# =============================== Dataset ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - PRED_LEN + 1

    def __getitem__(self, idx: int):
        # Input: all features of the previous seq_len rows
        x = self.data[idx: idx + self.seq_len, :]
        # Target: close value at position seq_len (index 3)
        y = self.data[idx + self.seq_len, 3]
        # Closing price at the current moment, used to calculate TrendAcc
        previous_close = self.data[idx + self.seq_len - 1, 3]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
            torch.tensor(previous_close, dtype=torch.float32)
        )


# =============================== Data Preprocessing ===============================
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

    # 4. Clean data
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # 5. Extract features
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

    # Extract timestamps
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


# =============================== Core Model: Simple DLinear ===============================

class MovingAvg(nn.Module):
    """
    Moving average module, used to decompose trend component
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        # Padding both ends to keep sequence length consistent
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # AvgPool1d requires input [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition module: decomposes series into Trend and Seasonal (residual/seasonal) components
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # Extract trend
        res = x - moving_mean  # Extract residual (Seasonal)
        return res, moving_mean


class SimpleDLinear(nn.Module):
    """
    Pure DLinear model implementation
    """

    def __init__(self, seq_len: int, pred_len: int, in_features: int, kernel_size: int = 25):
        super(SimpleDLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 1. Decomposition module
        # kernel_size is recommended to be odd, e.g., 25
        self.decompsition = SeriesDecomp(kernel_size)

        # 2. Linear layers
        # We flatten all input features (seq_len * in_features), then map to prediction results
        # This is a simple and effective method to handle multivariate input for univariate (or multivariate) prediction

        # Linear mapping for seasonal/residual component
        self.linear_seasonal = nn.Linear(seq_len * in_features, pred_len)

        # Linear mapping for trend component
        self.linear_trend = nn.Linear(seq_len * in_features, pred_len)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        # DLinear usually doesn't need complex initialization, but proper initialization helps convergence
        nn.init.xavier_uniform_(self.linear_seasonal.weight)
        nn.init.xavier_uniform_(self.linear_trend.weight)
        nn.init.zeros_(self.linear_seasonal.bias)
        nn.init.zeros_(self.linear_trend.bias)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, In_Features]

        # 1. Decomposition
        seasonal_init, trend_init = self.decompsition(x)

        # 2. Flatten input (Batch, Seq_Len * In_Features)
        # This allows the model to utilize correlations between different features simultaneously
        seasonal_init = seasonal_init.reshape(seasonal_init.shape[0], -1)
        trend_init = trend_init.reshape(trend_init.shape[0], -1)

        # 3. Linear mapping
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)

        # 4. Combine results
        x = seasonal_output + trend_output

        # Output shape: [Batch, Pred_Len] -> We need [Batch, 1] because PRED_LEN=1
        return x


# =============================== Training and Utility Functions ===============================
def denormalize_close(scaled_val, scaler, num_features):
    dummy = np.zeros((1, num_features))
    dummy[0, 3] = scaled_val  # Close is at index 3
    denorm = scaler.inverse_transform(dummy)[0, 3]
    return np.exp(denorm) - 1e-8


def run_process():
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 1. Prepare data
    train_loader, test_loader, scaler, test_target_dates, num_features = preprocess_data(df, LOOK_BACK)
    print(f"Data prepared. Using {num_features} features. Test samples: {len(test_target_dates)}")

    # 2. Initialize model (replace with SimpleDLinear)
    # kernel_size=25 is about one month of trading days, suitable for daily-level trend extraction
    model = SimpleDLinear(seq_len=LOOK_BACK, pred_len=PRED_LEN, in_features=num_features, kernel_size=21).to(device)

    # 3. Training configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()  # DLinear usually uses MSE or L1 loss

    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    print("Starting training (Simple DLinear)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            # by shape is [Batch, 1], pred is also [Batch, 1]
            loss = criterion(pred, by)
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
                val_loss += criterion(pred, by).item()
        val_loss /= len(test_loader)

        # Learning rate adjustment
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:  # Simple models may need longer patience
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 4. Evaluation
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

    # Calculate trend accuracy
    true_trend = np.sign(final_trues_np - final_prev_closes_np)
    pred_trend = np.sign(final_preds_np - final_prev_closes_np)
    trend_acc = np.mean(true_trend == pred_trend) * 100

    print("\n" + "=" * 40)
    print(" Evaluation Metrics (Test Set - DLinear) ")
    print("=" * 40)
    print(f" MAE          : {mae:.4f}")
    print(f" RMSE         : {rmse:.4f}")
    print(f" MSE          : {mse:.4f}")
    print(f" R^2          : {r2:.4f}")
    print(f" Trend Acc    : {trend_acc:.2f}%")
    print("=" * 40 + "\n")

    # 7. Save file
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
WR
========================================
 Evaluation Metrics (Test Set - DLinear) 
========================================
 MAE          : 57.2549
 RMSE         : 81.0015
 MSE          : 6561.2449
 R^2          : 0.9476
 Trend Acc    : 49.20%
========================================

"""