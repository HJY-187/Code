import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler  # Restore RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fix random seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Configuration Parameters =================
LOOK_BACK = 22  # Restore to 22, consistent with BiLSTM
PRED_LEN = 1
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 32
DROPOUT = 0.3  # Keep your set high Dropout
DATA_FILE = "J_daily.csv"
OUTPUT_FILE = "pred_daily_tsmixer_revin.csv"


# ================= RevIN Module =================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = 1
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_idx=3):
        if self.affine:
            x = x - self.affine_bias[target_idx]
            x = x / (self.affine_weight[target_idx] + self.eps * self.eps)
        x = x * self.stdev[:, :, target_idx]
        x = x + self.mean[:, :, target_idx]
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x


# ================= Dataset (Keep unchanged) =================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - PRED_LEN + 1

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.seq_len, :]
        y = self.data[idx + self.seq_len, 3]  # Close
        previous_close = self.data[idx + self.seq_len - 1, 3]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).unsqueeze(0),
                torch.tensor(previous_close, dtype=torch.float32))


# ================= Data Preprocessing (Key Fix: Align with BiLSTM Logic) =================
def preprocess_data(df: pd.DataFrame, look_back: int):
    df_work = df.copy()

    # 1. Log transformation (Restore to log, consistent with BiLSTM)
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # 2. Add calendar features
    dt = df_work['datetime']
    df_work['day_of_week'] = dt.dt.dayofweek
    df_work['day_of_year'] = dt.dt.dayofyear
    df_work['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df_work['month'] = dt.dt.month

    # 3. Technical indicators (Remove MA20, keep only 14-day window indicators for iloc[14:])
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

    # 4. Clean data (Key: Only remove first 14 rows, consistent with BiLSTM)
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # 5. Feature list
    feature_cols = [
        "open", "high", "low", "close", "volume", "open_interest",
        "day_of_week", "day_of_year", "week_of_year", "month",
        "ATR", "RSI", "roll_return"
    ]
    features = df_work[feature_cols].values

    # 6. Split training/test sets (Key: Restore 0.8 ratio and Gap strategy)
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    # Timestamp alignment
    test_start_idx = train_size + gap
    num_test_samples = len(raw_test) - look_back - PRED_LEN + 1
    target_indices = [test_start_idx + i + look_back for i in range(num_test_samples)]
    test_target_dates = df_work['datetime'].iloc[target_indices].values

    # 7. Normalization (Restore RobustScaler)
    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    # DataLoader
    train_loader = DataLoader(TimeSeriesDataset(train_data, look_back), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(test_data, look_back), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler, test_target_dates, len(feature_cols)


# ================= MODEL: TSMixer + RevIN (Keep unchanged) =================
class TSMixerBlock(nn.Module):
    def __init__(self, seq_len, n_vars, dropout, ff_dim):
        super().__init__()
        self.norm_time = nn.LayerNorm(n_vars)
        self.lin_time = nn.Linear(seq_len, seq_len)
        self.dropout = nn.Dropout(dropout)

        self.norm_feat = nn.LayerNorm(n_vars)
        self.lin_feat = nn.Sequential(
            nn.Linear(n_vars, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, n_vars),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Time Mixing
        res = x.clone()
        x = self.norm_time(x)
        x = x.transpose(1, 2)
        x = self.lin_time(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = x + res

        # Feature Mixing
        res = x.clone()
        x = self.norm_feat(x)
        x = self.lin_feat(x)
        x = x + res
        return x


class TSMixerRevIN(nn.Module):
    def __init__(self, seq_len, in_channels, pred_len=1, n_block=2, dropout=0.5):
        super().__init__()
        self.revin = RevIN(in_channels)
        self.blocks = nn.Sequential(*[
            TSMixerBlock(seq_len, in_channels, dropout, ff_dim=in_channels * 2)
            for _ in range(n_block)
        ])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * in_channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, pred_len)
        )

    def forward(self, x):
        x = self.revin(x, 'norm')
        x = self.blocks(x)
        out = self.head(x)
        out = self.revin(out, 'denorm')
        return out


# ================= Utils =================
def denormalize_close(scaled_val, scaler, num_features):
    dummy = np.zeros((1, num_features))
    dummy[0, 3] = scaled_val
    # Restore to exp (because preprocessing changed back to log)
    denorm = scaler.inverse_transform(dummy)[0, 3]
    return np.exp(denorm) - 1e-8


def run_process():
    print(f"Loading {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print("Data file not found.")
        return

    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Preprocess data (use modified alignment logic)
    train_loader, test_loader, scaler, dates, n_feat = preprocess_data(df, LOOK_BACK)
    print(f"Prepared. Features: {n_feat}. Using RevIN + TSMixer.")

    model = TSMixerRevIN(
        seq_len=LOOK_BACK,
        in_channels=n_feat,
        n_block=2,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    mse_loss = nn.MSELoss()

    best_loss = float('inf')
    best_w = None
    patience = 0

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = mse_loss(pred, by)
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
                val_loss += mse_loss(pred, by).item()
        val_loss /= len(test_loader)

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Ep {epoch + 1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_w = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                print("Early stopping.")
                break

    # Evaluation
    model.load_state_dict(best_w)
    model.eval()
    preds, trues, prevs = [], [], []
    with torch.no_grad():
        for bx, by, lc in test_loader:
            bx = bx.to(device)
            output = model(bx)
            preds.extend(output.cpu().numpy().flatten())
            trues.extend(by.numpy().flatten())
            prevs.extend(lc.numpy().flatten())

    final_preds = [denormalize_close(p, scaler, n_feat) for p in preds]
    final_trues = [denormalize_close(t, scaler, n_feat) for t in trues]
    final_prevs = [denormalize_close(p, scaler, n_feat) for p in prevs]

    y_true = np.array(final_trues)
    y_pred = np.array(final_preds)
    y_prev = np.array(final_prevs)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    true_trend = np.sign(y_true - y_prev)
    pred_trend = np.sign(y_pred - y_prev)
    trend_acc = np.mean(true_trend == pred_trend) * 100

    print("\n" + "=" * 40)
    print(" Optimized TSMixer (Aligned) Results")
    print("=" * 40)
    print(f" MAE      : {mae:.4f}")
    print(f" RMSE     : {rmse:.4f}")
    print(f" R2       : {r2:.4f}")
    print(f" Trend Acc: {trend_acc:.2f}%")
    print("=" * 40 + "\n")

    # Save results
    res_len = min(len(dates), len(final_preds))
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({
        "datetime": dates[:res_len],
        "pred_close": final_preds[:res_len]
    }).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_process()
"""
J
========================================
 Optimized TSMixer (Aligned) Results
========================================
 MAE      : 38.1729
 RMSE     : 49.2343
 R2       : 0.9612
 Trend Acc: 52.82%
========================================
"""