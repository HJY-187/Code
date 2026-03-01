import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRED_LEN = 12
BATCH_SIZE = 64
EPOCHS = 160
LR = 5e-4


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len, :9]          # 9个特征
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, 3]
        current = self.data[idx + self.seq_len - 1, 3]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(current, dtype=torch.float32),
        )


def preprocess_data(df: pd.DataFrame, look_back: int, pred_len: int):
    df_work = df.copy()

    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    df_work['TR'] = np.maximum.reduce([
        df_work['high'] - df_work['low'],
        np.abs(df_work['high'] - df_work['close'].shift(1)),
        np.abs(df_work['low'] - df_work['close'].shift(1))
    ])
    df_work['ATR'] = df_work['TR'].rolling(14).mean()

    delta = df_work['close'] - df_work['close'].shift(1)
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, 1e-8)
    df_work['RSI'] = 100 - 100 / (1 + avg_gain / avg_loss)

    df_work['roll_return'] = (df_work['close'] - df_work['close'].shift(5)) / df_work['close'].shift(5)

    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    feature_cols = [
        "open", "high", "low", "close", "volume", "open_interest",
        "ATR", "RSI", "roll_return"
    ]
    features = df_work[feature_cols].values
    dates = df_work["datetime"].values

    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_d = scaler.transform(raw_train)
    test_d = scaler.transform(raw_test)

    train_ds = TimeSeriesDataset(train_d, look_back, pred_len)
    test_ds = TimeSeriesDataset(test_d, look_back, pred_len)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler, dates, train_size, look_back


class PMain(nn.Module):

    def __init__(self, seq_len: int, pred_len: int, in_features: int = 9):
        super().__init__()
        self.in_features = in_features

        self.feature_weights = nn.Parameter(torch.ones(in_features))

        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.residual_linear = nn.Linear(seq_len, pred_len)

        self.decoder = nn.Sequential(
            nn.Linear(pred_len, pred_len * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(pred_len * 2, pred_len)
        )

    def moving_avg(self, x, kernel=5):
        p = (kernel - 1) // 2
        avg = nn.AvgPool1d(kernel, 1, p)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):               # [B, L, F]
        w = self.feature_weights.unsqueeze(0).unsqueeze(0)
        xw = x * w

        trend = self.moving_avg(xw, kernel=5)
        resid = xw - trend

        t_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        r_pred = self.residual_linear(resid.permute(0, 2, 1)).permute(0, 2, 1)

        base = (t_pred + r_pred)[:, :, 3]       # close

        refined = base + 0.08 * self.decoder(base)

        return refined


class DirectionalLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.SmoothL1Loss()

    def forward(self, pred, true, last):
        loss_num = self.mse(pred, true)
        last = last.view(-1, 1)
        true_d = true - last
        pred_d = pred - last
        penalty = F.relu(-true_d * pred_d)
        return loss_num + self.alpha * penalty.mean()


def denormalize_multi(scaled: np.ndarray, scaler):
    n, p = scaled.shape
    out = np.zeros((n, p))
    for i in range(p):
        dummy = np.zeros((n, 9))
        dummy[:, 3] = scaled[:, i]
        out[:, i] = np.exp(scaler.inverse_transform(dummy)[:, 3]) - 1e-8
    return out


def main(data_file="I_5.csv", output_file="../strategy2/pred_minute_advanced.csv"):
    if not os.path.exists(data_file):
        print(f"not founf：{data_file}")
        return

    df = pd.read_csv(data_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    look_back = 48 if "30" in os.path.basename(data_file) else 20
    print(f"use look_back = {look_back}")

    train_loader, test_loader, scaler, dates, train_size, lb = preprocess_data(df, look_back, PRED_LEN)

    model = PMain(lb, PRED_LEN, in_features=9).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=8)
    crit = DirectionalLoss(alpha=1.0)

    best_loss = float('inf')
    best_w = None
    patience = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = 0
        for x, y, last in train_loader:
            x, y, last = x.to(device), y.to(device), last.to(device)
            opt.zero_grad()
            p = model(x)
            loss = crit(p, y, last)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            tl += loss.item()
        tl /= len(train_loader)

        model.eval()
        vl = 0
        with torch.no_grad():
            for x, y, last in test_loader:
                x, y, last = x.to(device), y.to(device), last.to(device)
                p = model(x)
                vl += crit(p, y, last).item()
        vl /= len(test_loader)

        sch.step(vl)

        if ep % 20 == 0:
            print(f"[{ep:3d}] train:{tl:.6f}  val:{vl:.6f}")

        if vl < best_loss:
            best_loss = vl
            best_w = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                print("early stop")
                break

    model.load_state_dict(best_w)
    model.eval()

    ps, ts = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            p = model(x).cpu().numpy()
            ps.append(p)
            ts.append(y.numpy())

    ps = np.concatenate(ps, axis=0)
    ts = np.concatenate(ts, axis=0)

    pred_price = denormalize_multi(ps, scaler)
    true_price = denormalize_multi(ts, scaler)

    start_idx = train_size + 2 * look_back
    res_dates = dates[start_idx : start_idx + len(pred_price)]

    print("\n" + "="*70)
    print("Step 1–12")
    for step in range(PRED_LEN):
        t = true_price[:, step]
        p = pred_price[:, step]
        mse = mean_squared_error(t, p)
        mae = mean_absolute_error(t, p)
        rmse = math.sqrt(mse)
        r2 = r2_score(t, p)
        print(f"Step {step+1:2d}   MSE:{mse: .6f}  MAE:{mae: .6f}  RMSE:{rmse: .6f}  R²:{r2: .4f}")

    print("\nmetrics")
    mse = mean_squared_error(true_price, pred_price)
    mae = mean_absolute_error(true_price, pred_price)
    rmse = math.sqrt(mse)
    r2 = r2_score(true_price, pred_price)
    print(f"MSE:{mse:.6f}  MAE:{mae:.6f}  RMSE:{rmse:.6f}  R²:{r2:.4f}")
    print("="*70)

    # save
    df_out = pd.DataFrame({"datetime": res_dates})
    for k in range(PRED_LEN):
        df_out[f"pred_step_{k+1}"] = pred_price[:, k]
    df_out["pred_avg"] = pred_price.mean(axis=1)

    df_out.to_csv(output_file, index=False, encoding="utf-8")
    print(f"saved:{output_file}")


if __name__ == "__main__":
    main()
"""

[ 20] train:0.001624  val:0.001891
[ 40] train:0.001461  val:0.001691
[ 60] train:0.001429  val:0.001646
[ 80] train:0.001408  val:0.001665
[100] train:0.001402  val:0.001667
[120] train:0.001392  val:0.001598
[140] train:0.001394  val:0.001591
[160] train:0.001388  val:0.001596

======================================================================
Step 1–12
Step  1   MSE: 4.559944  MAE: 1.415316  RMSE: 2.135403  R²: 0.9971
Step  2   MSE: 8.292281  MAE: 1.937299  RMSE: 2.879632  R²: 0.9947
Step  3   MSE: 11.827146  MAE: 2.320604  RMSE: 3.439062  R²: 0.9925
Step  4   MSE: 15.218391  MAE: 2.641506  RMSE: 3.901076  R²: 0.9903
Step  5   MSE: 18.514192  MAE: 2.929159  RMSE: 4.302812  R²: 0.9882
Step  6   MSE: 21.843673  MAE: 3.207199  RMSE: 4.673722  R²: 0.9861
Step  7   MSE: 25.244230  MAE: 3.464748  RMSE: 5.024364  R²: 0.9840
Step  8   MSE: 28.680393  MAE: 3.716857  RMSE: 5.355408  R²: 0.9818
Step  9   MSE: 32.096939  MAE: 3.964490  RMSE: 5.665416  R²: 0.9796
Step 10   MSE: 35.404492  MAE: 4.205241  RMSE: 5.950167  R²: 0.9775
Step 11   MSE: 38.515554  MAE: 4.416584  RMSE: 6.206090  R²: 0.9755
Step 12   MSE: 41.680550  MAE: 4.605640  RMSE: 6.456048  R²: 0.9735

MSE:23.489815  MAE:3.235387  RMSE:4.846629  R²:0.9851
======================================================================
"""
