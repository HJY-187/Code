# -*- coding: utf-8 -*-
"""
优化版 - 精简版 DLinear 主路径
（移除 N-HiTS 和 Autoformer-lite 两个辅助分支，仅保留 DLinear 主干 + 特征加权 + decoder 精修）
目标：Next Day Close | 输出：../策略/pred_daily_advanced.csv
"""

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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================== 配置参数 ==============================
LOOK_BACK = 22
PRED_LEN = 1
EPOCHS = 180
LR = 5e-4
BATCH_SIZE = 64
DATA_FILE = "I_daily.csv"
OUTPUT_FILE = "../策略2/pred_daily_advanced.csv"


# ============================== Dataset ==============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - PRED_LEN + 1

    def __getitem__(self, idx: int):
        # 输入：前 seq_len 行的所有特征
        x = self.data[idx: idx + self.seq_len, :]

        # 目标：下一交易日的 close 值 (索引3)
        y = self.data[idx + self.seq_len, 3]

        # 当前时刻的收盘价（用于趋势准确率计算）
        previous_close = self.data[idx + self.seq_len - 1, 3]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
            torch.tensor(previous_close, dtype=torch.float32)
        )


# ============================== 数据预处理 ==============================
def preprocess_data(df: pd.DataFrame, look_back: int):
    df_work = df.copy()

    # 1. 对数变换（价格类字段）
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # 2. 增加日历特征
    dt = df_work['datetime']
    df_work['day_of_week'] = dt.dt.dayofweek
    df_work['day_of_year'] = dt.dt.dayofyear
    df_work['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df_work['month'] = dt.dt.month

    # 3. 技术指标
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

    # 4. 清洗
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # 5. 特征列
    feature_cols = [
        "open", "high", "low", "close", "volume", "open_interest",
        "day_of_week", "day_of_year", "week_of_year", "month",
        "ATR", "RSI", "roll_return"
    ]
    features = df_work[feature_cols].values

    # 6. 训练/测试划分
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    # 测试集目标日期
    test_start_idx = train_size + gap
    num_test_samples = len(raw_test) - look_back - PRED_LEN + 1
    target_indices = [test_start_idx + i + look_back for i in range(num_test_samples)]
    test_target_dates = df_work['datetime'].iloc[target_indices].values

    # 7. 归一化
    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    # 8. Dataset & DataLoader
    train_ds = TimeSeriesDataset(train_data, seq_len=look_back)
    test_ds = TimeSeriesDataset(test_data, seq_len=look_back)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler, test_target_dates, len(feature_cols)


# ============================== 模型定义（仅保留主路径） ==============================
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation,
                                   groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        y = self.depthwise(x)
        y = F.gelu(y)
        y = self.pointwise(y)
        y = self.bn(y)
        return y


class PMain(nn.Module):
    """
    精简版 DLinear 主路径：
    • 特征加权
    • 移动平均分解 trend / residual
    • 分别线性投影
    • 融合 + 小型 MLP 精修
    """
    def __init__(self, seq_len: int, in_features: int, ma_kernel_size: int = 21):
        super().__init__()
        self.in_features = in_features
        self.ma_kernel_size = ma_kernel_size

        # 可学习的特征重要性权重
        self.feature_weights = nn.Parameter(torch.ones(in_features))

        # 趋势 & 残差 投影
        self.trend_linear = nn.Linear(seq_len, PRED_LEN)
        self.residual_linear = nn.Linear(seq_len, PRED_LEN)

        # 最终精修 MLP
        self.decoder = nn.Sequential(
            nn.Linear(PRED_LEN, PRED_LEN * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(PRED_LEN * 2, PRED_LEN)
        )

    def moving_avg(self, x):
        # x: [B, L, C] → [B, C, L]
        padding = (self.ma_kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=self.ma_kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):           # x: [B, L, in_features]
        # 特征加权
        w = self.feature_weights.unsqueeze(0).unsqueeze(0)     # [1,1,C]
        weighted_x = x * w

        # 经典 DLinear 分解
        trend = self.moving_avg(weighted_x)
        residual = weighted_x - trend

        # 分别投影到预测长度
        trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)     # [B, P, C]
        residual_pred = self.residual_linear(residual.permute(0, 2, 1)).permute(0, 2, 1)

        # 只取 close 通道的预测（索引 3）
        base_pred = (trend_pred + residual_pred)[:, :, 3]       # [B, P]

        # 可选的精修层（通常效果更好）
        refined = base_pred + 0.1 * self.decoder(base_pred)

        return refined


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

        # 方向错误惩罚
        penalty = F.relu(-1.0 * true_diff * pred_diff)

        return num_loss + self.alpha * torch.mean(penalty)


# ============================== 工具函数 ==============================
def denormalize_close(scaled_val, scaler, num_features):
    dummy = np.zeros((1, num_features))
    dummy[0, 3] = scaled_val
    denorm = scaler.inverse_transform(dummy)[0, 3]
    return np.exp(denorm) - 1e-8


# ============================== 主流程 ==============================
def run_process():
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 数据准备
    train_loader, test_loader, scaler, test_target_dates, num_features = preprocess_data(df, LOOK_BACK)
    print(f"Data prepared. Features: {num_features}, Test samples: {len(test_target_dates)}")

    # 模型
    model = PMain(
        seq_len=LOOK_BACK,
        in_features=num_features,
        ma_kernel_size=21
    ).to(device)

    # 优化器 & 调度
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    criterion = DirectionalLoss(alpha=2.0)

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

        # 验证
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
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型
    print("Training finished. Evaluating best model...")
    model.load_state_dict(best_model_weights)
    model.eval()

    preds_scaled, trues_scaled, prev_closes_scaled = [], [], []

    with torch.no_grad():
        for bx, by, b_prev in test_loader:
            bx = bx.to(device)
            output = model(bx)
            preds_scaled.extend(output.cpu().numpy().flatten())
            trues_scaled.extend(by.numpy().flatten())
            prev_closes_scaled.extend(b_prev.numpy().flatten())

    # 反归一化
    final_preds, final_trues, final_prev_closes = [], [], []
    for p, t, pc in zip(preds_scaled, trues_scaled, prev_closes_scaled):
        final_preds.append(denormalize_close(p, scaler, num_features))
        final_trues.append(denormalize_close(t, scaler, num_features))
        final_prev_closes.append(denormalize_close(pc, scaler, num_features))

    # 评估指标
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

    print("\n" + "="*40)
    print(" Evaluation Metrics (Test Set) ")
    print("="*40)
    print(f" MAE       : {mae:.4f}")
    print(f" RMSE      : {rmse:.4f}")
    print(f" MSE       : {mse:.4f}")
    print(f" R²        : {r2:.4f}")
    print(f" Trend Acc : {trend_acc:.2f}%")
    print("="*40 + "\n")

    # 保存预测结果
    min_len = min(len(test_target_dates), len(final_preds))
    results_df = pd.DataFrame({
        "datetime": test_target_dates[:min_len],
        "pred_close": final_preds[:min_len]
    })
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_process()
