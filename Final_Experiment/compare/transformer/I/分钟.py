# -*- coding: utf-8 -*-
"""
Model: Simple Transformer
Description: 使用位置编码和Transformer Encoder提取特征。
"""

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


# Dataset & Preprocessing (Same as above, condensed for brevity but functional)
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data, self.seq_len, self.pred_len = data, seq_len, pred_len

    def __len__(self): return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx + self.seq_len, :6], dtype=torch.float32),
                torch.tensor(self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, 3], dtype=torch.float32),
                torch.tensor(self.data[idx + self.seq_len - 1, 3], dtype=torch.float32))


def preprocess_and_split_data(df, look_back, pred_len):
    df = df.copy()
    for c in ["open", "high", "low", "close"]: df[c] = np.log(df[c] + 1e-8)
    # Features (simplified logic from template)
    df = df.iloc[14:].dropna().reset_index(drop=True)  # Assuming indicators calculated or skipping for simplicity
    # For full compatibility, reusing the robust features requires the full calculation block
    # Here we assume the input DF has basic cols. Adding minimal required indicator placeholders
    if 'ATR' not in df.columns: df['ATR'], df['RSI'], df['roll_return'] = 0, 0, 0

    features = df[["open", "high", "low", "close", "volume", "open_interest", "ATR", "RSI", "roll_return"]].values
    scaler = RobustScaler().fit(features[:int(len(features) * 0.8)])
    train_data = scaler.transform(features[:int(len(features) * 0.8)])
    test_data = scaler.transform(features[int(len(features) * 0.8) + look_back:])

    train_loader = DataLoader(TimeSeriesDataset(train_data, look_back, pred_len), batch_size=64, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(TimeSeriesDataset(test_data, look_back, pred_len), batch_size=64, shuffle=False,
                             num_workers=0)
    return train_loader, test_loader, scaler, df["datetime"].values, int(len(features) * 0.8), look_back


# =============================== 模型定义：Simple Transformer ===============================
class PositionalEncoding(nn.Module):
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


class TransformerModel(nn.Module):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=2, pred_len=12, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 取最后一个时间步的特征进行预测
        x = x[:, -1, :]
        output = self.decoder(x)
        return output


# =============================== Loss & Tools (Copied) ===============================
class DirectionalLoss(nn.Module):
    def __init__(self, alpha=1.0): super().__init__(); self.alpha = alpha; self.base = nn.SmoothL1Loss()

    def forward(self, p, t, l):
        return self.base(p, t) + self.alpha * torch.mean(F.relu(-1 * (t - l.view(-1, 1)) * (p - l.view(-1, 1))))


def denormalize_predictions(scaled_preds, scaler):
    n, p = scaled_preds.shape;
    original = np.zeros_like(scaled_preds)
    for i in range(p):
        dummy = np.zeros((n, 9));
        dummy[:, 3] = scaled_preds[:, i]
        original[:, i] = np.exp(scaler.inverse_transform(dummy)[:, 3]) - 1e-8
    return original


# =============================== Main ===============================
def train_and_eval():
    data_file = "I_5.csv"
    # 改动1：文件不存在时添加明确提示
    if not os.path.exists(data_file):
        print(f"错误：未找到数据文件 {data_file}，请检查文件路径是否正确！")
        return

    df = pd.read_csv(data_file)
    # Ensure full feature calculation (TR, ATR, etc) normally happens here
    # For brevity, assuming df is loaded.
    # Recalculate indicators to be safe:
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    df['TR'] = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    df['ATR'] = df['TR'].rolling(14).mean()
    df['roll_return'] = df['close'].pct_change(5)

    # 改动2：修复RSI计算中的除零错误（避免分母为0）
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    loss = loss.replace(0, 1e-8)  # 替换0值，防止除零
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    train_loader, test_loader, scaler, dates, t_size, lb = preprocess_and_split_data(df, 48, PRED_LEN)

    model = TransformerModel(input_dim=6, d_model=64, nhead=4, num_layers=2, pred_len=PRED_LEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    crit = DirectionalLoss()

    # 改动3：添加训练损失监控，每5轮打印一次平均损失
    print("开始训练 Simple Transformer 模型...")
    for ep in range(160):  # Reduced epochs for demo
        model.train()
        epoch_loss = 0.0  # 累加本轮所有batch的损失
        for bx, by, _ in train_loader:
            opt.zero_grad()
            pred = model(bx.to(device))
            loss = crit(pred, by.to(device), bx[:, -1, 3].to(device))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # 每5轮打印一次损失，方便监控收敛情况
        if (ep + 1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {ep + 1}/160 | 平均训练损失: {avg_loss:.6f}")
        else:
            print(f"Epoch {ep + 1}/160 complete")

    # Eval
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            preds.append(model(bx.to(device)).cpu().numpy())
            trues.append(by.numpy())

    preds = denormalize_predictions(np.concatenate(preds), scaler)
    trues = denormalize_predictions(np.concatenate(trues), scaler)

    # 改动4：完善评估指标，新增每步MAE/RMSE/R2 + 整体指标
    print("\n==================== Transformer 预测结果评估 ====================")
    # 每一步的详细指标
    step_metrics = []
    for i in range(PRED_LEN):
        mse = mean_squared_error(trues[:, i], preds[:, i])
        mae = mean_absolute_error(trues[:, i], preds[:, i])
        rmse = math.sqrt(mse)
        r2 = r2_score(trues[:, i], preds[:, i])
        step_metrics.append([mse, mae, rmse, r2])
        print(f"第 {i + 1:02d} 步 | MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

    # 计算整体指标（所有步合并）
    overall_mse = mean_squared_error(trues, preds)
    overall_mae = mean_absolute_error(trues, preds)
    overall_rmse = math.sqrt(overall_mse)
    overall_r2 = r2_score(trues, preds)

    print("\n==================== 整体评估指标（12步） ====================")
    print(f"整体 MSE: {overall_mse:.6f}")
    print(f"整体 MAE: {overall_mae:.6f}")
    print(f"整体 RMSE: {overall_rmse:.6f}")
    print(f"整体 R2: {overall_r2:.6f}")

    # 改动5：创建保存目录，避免路径不存在报错
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pred_transformer.csv")
    pd.DataFrame(preds).to_csv(save_path, index=False)
    print(f"\n预测结果已保存至: {save_path}")


if __name__ == "__main__":
    train_and_eval()

"""
==================== Transformer 预测结果评估 ====================
第 01 步 | MSE: 7.007747, MAE: 1.865906, RMSE: 2.647215, R2: 0.995533
第 02 步 | MSE: 11.595783, MAE: 2.399111, RMSE: 3.405258, R2: 0.992607
第 03 步 | MSE: 15.555811, MAE: 2.798168, RMSE: 3.944086, R2: 0.990080
第 04 步 | MSE: 20.535713, MAE: 3.228841, RMSE: 4.531635, R2: 0.986901
第 05 步 | MSE: 27.910418, MAE: 3.832172, RMSE: 5.283031, R2: 0.982192
第 06 步 | MSE: 36.838890, MAE: 4.410539, RMSE: 6.069505, R2: 0.976489
第 07 步 | MSE: 44.039139, MAE: 4.799526, RMSE: 6.636199, R2: 0.971886
第 08 步 | MSE: 55.905785, MAE: 5.400020, RMSE: 7.477017, R2: 0.964299
第 09 步 | MSE: 62.722565, MAE: 5.724292, RMSE: 7.919758, R2: 0.959935
第 10 步 | MSE: 70.908592, MAE: 6.052826, RMSE: 8.420724, R2: 0.954691
第 11 步 | MSE: 77.285095, MAE: 6.344962, RMSE: 8.791194, R2: 0.950598
第 12 步 | MSE: 88.466484, MAE: 6.780204, RMSE: 9.405662, R2: 0.943430

==================== 整体评估指标（12步） ====================
整体 MSE: 43.230972
整体 MAE: 4.469714
整体 RMSE: 6.575026
整体 R2: 0.972387

"""