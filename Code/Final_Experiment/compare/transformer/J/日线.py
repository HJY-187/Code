# -- coding: utf-8 --
"""
日线策略 - 简单Transformer模型 (含MAE/RMSE/R2/Trend Acc)
"""
import os, math, copy
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

# 参数
LOOK_BACK = 22
PRED_LEN = 1
EPOCHS = 180
LR = 5e-4
BATCH_SIZE = 64
DATA_FILE = "J_daily.csv"
OUTPUT_FILE = "pred_daily_transformer.csv"


# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self): return len(self.data) - self.seq_len - PRED_LEN + 1

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.float32),
                torch.tensor(self.data[idx + self.seq_len, 3], dtype=torch.float32).unsqueeze(0),
                torch.tensor(self.data[idx + self.seq_len - 1, 3], dtype=torch.float32))


# 预处理
def preprocess_data(df: pd.DataFrame, look_back: int):
    df_work = df.copy()
    for col in ["open", "high", "low", "close"]: df_work[col] = np.log(df_work[col] + 1e-8)
    dt = df_work['datetime']
    df_work['day_of_week'] = dt.dt.dayofweek
    df_work['day_of_year'] = dt.dt.dayofyear
    df_work['month'] = dt.dt.month
    df_work['TR'] = np.maximum(df_work['high'] - df_work['low'], np.abs(df_work['close'].shift(1) - df_work['high']))
    df_work['ATR'] = df_work['TR'].rolling(14).mean()
    df_work['roll_return'] = (df_work['close'] - df_work['close'].shift(5)) / df_work['close'].shift(5)
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    feature_cols = ["open", "high", "low", "close", "volume", "open_interest", "day_of_week", "day_of_year", "month",
                    "ATR", "roll_return"]
    features = df_work[feature_cols].values
    train_size = int(len(features) * 0.8)

    scaler = RobustScaler()
    train_data = scaler.fit_transform(features[:train_size])
    test_data = scaler.transform(features[train_size + look_back:])

    target_dates = df_work['datetime'].iloc[train_size + look_back * 2:].values[:len(test_data) - look_back]

    return (DataLoader(TimeSeriesDataset(train_data, look_back), batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(TimeSeriesDataset(test_data, look_back), batch_size=BATCH_SIZE, shuffle=False),
            scaler, target_dates, len(feature_cols))


# ================= MODEL: Transformer =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, seq_len, in_features, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model * seq_len, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.flatten(1)
        return self.decoder(x)


# 辅助函数
class DirectionalLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha;
        self.base = nn.SmoothL1Loss()

    def forward(self, p, t, lc):
        return self.base(p, t) + self.alpha * torch.mean(torch.nn.functional.relu(-1.0 * (t - lc) * (p - lc)))


def denorm(val, s, n):
    d = np.zeros((1, n));
    d[0, 3] = val
    return np.exp(s.inverse_transform(d)[0, 3]) - 1e-8


# 主流程
def run_process():
    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    tr_l, te_l, sc, dates, n_ft = preprocess_data(df, LOOK_BACK)

    model = TransformerModel(LOOK_BACK, n_ft).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8)
    crit = DirectionalLoss()

    best_loss = float('inf')
    best_st = None

    for ep in range(EPOCHS):
        model.train()
        l_sum = 0
        for x, y, lc in tr_l:
            opt.zero_grad()
            l = crit(model(x.to(device)), y.to(device), lc.to(device))
            l.backward();
            opt.step();
            l_sum += l.item()

        model.eval()
        v_sum = 0
        with torch.no_grad():
            for x, y, lc in te_l:
                v_sum += crit(model(x.to(device)), y.to(device), lc.to(device)).item()

        l_sum /= len(tr_l);
        v_sum /= len(te_l)
        sch.step(v_sum)
        if v_sum < best_loss:
            best_loss = v_sum;
            best_st = copy.deepcopy(model.state_dict())

        if (ep + 1) % 20 == 0: print(f"Ep {ep + 1} | T: {l_sum:.5f} V: {v_sum:.5f}")

    model.load_state_dict(best_st)
    preds, trues, prevs = [], [], []  # 新增：收集前一日收盘价
    with torch.no_grad():
        for x, y, lc in te_l:
            preds.extend(model(x.to(device)).cpu().numpy().flatten())
            trues.extend(y.numpy().flatten())
            prevs.extend(lc.numpy().flatten())  # 新增：收集前一日收盘价

    final_p = np.array([denorm(p, sc, n_ft) for p in preds])
    final_t = np.array([denorm(t, sc, n_ft) for t in trues])
    final_prevs = np.array([denorm(p, sc, n_ft) for p in prevs])  # 新增：反归一化前一日收盘价

    # 指标计算
    mae = mean_absolute_error(final_t, final_p)
    rmse = np.sqrt(mean_squared_error(final_t, final_p))
    r2 = r2_score(final_t, final_p)
    # 新增：计算趋势准确率
    trend_acc = np.mean(np.sign(final_t - final_prevs) == np.sign(final_p - final_prevs)) * 100

    print("\n" + "=" * 40)
    print(" Transformer Daily Evaluation Results")
    print("=" * 40)
    print(f" MAE      : {mae:.4f}")
    print(f" RMSE     : {rmse:.4f}")
    print(f" R2       : {r2:.4f}")
    print(f" Trend Acc: {trend_acc:.2f}%")  # 新增：打印趋势准确率
    print("=" * 40 + "\n")

    pd.DataFrame({"datetime": dates[:len(final_p)], "pred": final_p}).to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    run_process()

"""
========================================
 Transformer Daily Evaluation Results
========================================
 MAE      : 39.4239
 RMSE     : 50.1745
 R2       : 0.9597
 Trend Acc: 51.47%
========================================
"""