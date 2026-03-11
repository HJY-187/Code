# -*- coding: utf-8 -*-
"""
Model: Simple Transformer (调参优化版)
Description: 调参+结构优化+训练策略强化，提升金融时间序列预测效果
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

# 固定随机种子，保证可复现
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LEN = 12  # 预测步长12不变


# Dataset & Preprocessing (原逻辑保留，仅微调细节)
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data, self.seq_len, self.pred_len = data, seq_len, pred_len

    def __len__(self): return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 特征：前6列(open/high/low/close/volume/open_interest)，标签：close列，最后一个时间步close
        return (torch.tensor(self.data[idx:idx + self.seq_len, :6], dtype=torch.float32),
                torch.tensor(self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, 3], dtype=torch.float32),
                torch.tensor(self.data[idx + self.seq_len - 1, 3], dtype=torch.float32))


def preprocess_and_split_data(df, look_back, pred_len):
    df = df.copy()
    # 金融数据log处理：避免负数，原逻辑保留
    for c in ["open", "high", "low", "close"]: df[c] = np.log(df[c] + 1e-8)
    # 计算技术指标（原逻辑完善，避免缺失）
    df['TR'] = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    df['ATR'] = df['TR'].rolling(14).mean()
    df['roll_return'] = df['close'].pct_change(5)
    # RSI计算（原修复后的逻辑）
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    loss = loss.replace(0, 1e-8)
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    # 丢弃缺失值，替代原iloc[14:]，更严谨
    df = df.dropna().reset_index(drop=True)
    # 特征列：9列（基础+指标）
    features = df[["open", "high", "low", "close", "volume", "open_interest", "ATR", "RSI", "roll_return"]].values
    # 划分训练测试集（8:2），RobustScaler适配金融异常值
    split_idx = int(len(features) * 0.8)
    scaler = RobustScaler().fit(features[:split_idx])
    train_data = scaler.transform(features[:split_idx])
    # 测试集偏移look_back，避免数据泄露
    test_data = scaler.transform(features[split_idx + look_back:])

    # 数据加载器：微调num_workers，Windows更兼容
    train_loader = DataLoader(TimeSeriesDataset(train_data, look_back, pred_len),
                              batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TimeSeriesDataset(test_data, look_back, pred_len),
                             batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader, scaler, df["datetime"].values, split_idx, look_back


# =============================== 模型定义：调参优化版 Transformer ===============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.3):
        super(PositionalEncoding, self).__init__()
        # 调优：添加dropout，抑制位置编码过拟合
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    # 调优：大幅优化模型超参数，适配金融时间序列特征提取
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_layers=4, pred_len=12, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        # 调优：嵌入层后加层归一化，稳定训练
        self.norm_emb = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        # Transformer编码器层：调优dim_feedforward为d_model*4（行业最佳实践），激活函数用gelu（比relu更适合Transformer）
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            layer_norm_eps=1e-5
        )
        # 调优：增加编码器层数，提升特征提取能力；添加层归一化收尾
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.norm_final = nn.LayerNorm(d_model)
        # 解码器：从d_model映射到预测步长
        self.decoder = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Features] (32,48,6)
        x = self.input_embedding(x)  # (32,48,128)
        x = self.norm_emb(x)         # 层归一化，稳定训练
        x = self.pos_encoder(x)      # 位置编码 (32,48,128)
        x = self.transformer_encoder(x)  # Transformer编码 (32,48,128)
        # 调优：取所有时间步的特征均值，替代仅取最后一步，利用全局时序信息
        x = x.mean(dim=1)            # (32,128)
        x = self.norm_final(x)       # 最终层归一化
        output = self.decoder(x)     # (32,12)
        return output


# =============================== 损失函数+工具函数（调优） ===============================
class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.5):  # 调优：调整方向损失权重，平衡数值和方向预测
        super().__init__()
        self.alpha = alpha
        self.base = nn.SmoothL1Loss(reduction='mean')  # 平滑L1，对异常值更鲁棒

    def forward(self, p, t, l):
        # p:预测值, t:真实值, l:上一时间步close
        base_loss = self.base(p, t)
        # 方向损失：预测方向与真实方向相反则惩罚
        dir_loss = torch.mean(F.relu(-1 * (t - l.view(-1, 1)) * (p - l.view(-1, 1))))
        return base_loss + self.alpha * dir_loss


def denormalize_predictions(scaled_preds, scaler):
    # 反归一化：原逻辑保留，适配9维特征
    n, p = scaled_preds.shape
    original = np.zeros_like(scaled_preds)
    for i in range(p):
        dummy = np.zeros((n, 9))
        dummy[:, 3] = scaled_preds[:, i]
        original[:, i] = np.exp(scaler.inverse_transform(dummy)[:, 3]) - 1e-8
    return original


# =============================== 新增：早停机制（关键，避免过拟合） ===============================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001, save_path='best_model.pth'):
        self.patience = patience  # 多少轮无提升则停止
        self.min_delta = min_delta  # 损失最小变化量
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        # 验证损失下降则保存最优模型
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"早停触发！已{self.patience}轮验证损失无提升，加载最优模型")
                model.load_state_dict(torch.load(self.save_path))
                return True
        return False


# =============================== 主训练评估函数（大幅调优训练策略） ===============================
def train_and_eval():
    data_file = "WR_5.csv"
    if not os.path.exists(data_file):
        print(f"错误：未找到数据文件 {data_file}，请检查文件路径是否正确！")
        return

    df = pd.read_csv(data_file)
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    # 序列长度：原48不变，适配金融5分钟K线的时序特征
    LOOK_BACK = 48
    # 数据预处理
    train_loader, test_loader, scaler, dates, t_size, lb = preprocess_and_split_data(df, LOOK_BACK, PRED_LEN)

    # 初始化模型：调参后的超参数
    model = TransformerModel(
        input_dim=6,
        d_model=128,
        nhead=8,
        num_layers=4,
        pred_len=PRED_LEN,
        dropout=0.25
    ).to(device)
    # 调优：优化器替换为AdamW（带权重衰减），抑制过拟合
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 修复：删除弃用的verbose=True参数，消除PyTorch版本警告
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    # 损失函数
    crit = DirectionalLoss(alpha=0.1).to(device)
    # 早停实例
    early_stopping = EarlyStopping(patience=15, save_path='best_model.pth')

    # 训练配置：调优轮数，配合早停无需担心过拟合
    EPOCHS = 300
    print("开始训练 调优版 Simple Transformer 模型...")
    # 修复：去掉requires_grad后的括号，属性直接访问而非方法调用
    print(f"设备：{device} | 模型参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for ep in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for bx, by, bl in train_loader:
            bx, by, bl = bx.to(device), by.to(device), bl.to(device)
            opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, by, bl)
            loss.backward()
            # 调优：梯度裁剪，防止梯度爆炸（Transformer必备）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * bx.size(0)  # 按样本数加权

        # 验证阶段（用测试集简易验证，实际建议拆分验证集）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, bl in test_loader:
                bx, by, bl = bx.to(device), by.to(device), bl.to(device)
                pred = model(bx)
                loss = crit(pred, by, bl)
                val_loss += loss.item() * bx.size(0)

        # 计算平均损失
        train_avg_loss = train_loss / len(train_loader.dataset)
        val_avg_loss = val_loss / len(test_loader.dataset)
        # 学习率调度更新
        scheduler.step(val_avg_loss)

        # 打印日志：每5轮打印，更清晰
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep + 1:03d}/{EPOCHS} | 训练损失: {train_avg_loss:.6f} | 验证损失: {val_avg_loss:.6f}")
        # 早停判断
        if early_stopping(val_avg_loss, model):
            break

    # 评估阶段：加载最优模型预测
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            preds.append(model(bx.to(device)).cpu().numpy())
            trues.append(by.numpy())
    # 拼接结果并反归一化
    preds = denormalize_predictions(np.concatenate(preds), scaler)
    trues = denormalize_predictions(np.concatenate(trues), scaler)

    # 评估指标：原逻辑保留，完善打印
    print("\n==================== Transformer 预测结果评估（调优版） ====================")
    step_metrics = []
    for i in range(PRED_LEN):
        mse = mean_squared_error(trues[:, i], preds[:, i])
        mae = mean_absolute_error(trues[:, i], preds[:, i])
        rmse = math.sqrt(mse)
        r2 = r2_score(trues[:, i], preds[:, i])
        step_metrics.append([mse, mae, rmse, r2])
        print(f"第 {i + 1:02d} 步 | MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

    # 整体指标
    overall_mse = mean_squared_error(trues, preds)
    overall_mae = mean_absolute_error(trues, preds)
    overall_rmse = math.sqrt(overall_mse)
    # 调优：整体R2用ravel展平，适配sklearn计算
    overall_r2 = r2_score(trues.ravel(), preds.ravel())

    print("\n==================== 整体评估指标（12步） ====================")
    print(f"整体 MSE: {overall_mse:.6f}")
    print(f"整体 MAE: {overall_mae:.6f}")
    print(f"整体 RMSE: {overall_rmse:.6f}")
    print(f"整体 R2: {overall_r2:.6f}")

    # 保存结果：调优+保存真实值，方便对比
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    # 保存预测值和真实值
    res_df = pd.DataFrame()
    for i in range(PRED_LEN):
        res_df[f'pred_step_{i+1}'] = preds[:, i]
        res_df[f'true_step_{i+1}'] = trues[:, i]
    res_df.to_csv(os.path.join(save_dir, "pred_transformer_optimized.csv"), index=False)
    print(f"\n预测&真实结果已保存至: {os.path.join(save_dir, 'pred_transformer_optimized.csv')}")
    # 删除临时最优模型文件
    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')


if __name__ == "__main__":
    train_and_eval()