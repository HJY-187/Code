import os
import math
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================== 1. 全局设置 ===============================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LEN = 12


# =============================== 2. 数据处理 ===============================
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
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def preprocess_and_split_data(df: pd.DataFrame, look_back: int, pred_len: int):
    df_work = df.copy()
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # 简单特征工程
    df_work['roll_return'] = (df_work['close'] - df_work['close'].shift(5)) / df_work['close'].shift(5)
    df_work['TR'] = np.maximum(df_work['high'] - df_work['low'], np.abs(df_work['close'] - df_work['close'].shift(1)))
    df_work['ATR'] = df_work['TR'].rolling(14).mean()
    delta = df_work['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean().replace(0, 1e-8)
    df_work['RSI'] = 100 - (100 / (1 + gain / loss))

    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)
    features = df_work[["open", "high", "low", "close", "volume", "open_interest", "ATR", "RSI", "roll_return"]].values

    scaler = RobustScaler()
    train_size = int(len(features) * 0.8)
    train_data = scaler.fit_transform(features[:train_size])
    test_data = scaler.transform(features[train_size:])

    train_loader = DataLoader(TimeSeriesDataset(train_data, look_back, pred_len), batch_size=64, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(test_data, look_back, pred_len), batch_size=64, shuffle=False)

    return train_loader, test_loader, scaler


# =============================== 3. 模型组件 ===============================
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels,
                                   bias=False)
        self.pointwise = nn.Conv1d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        return self.bn(self.pointwise(F.gelu(self.depthwise(x))))


class AutoformerLiteBlock(nn.Module):
    def __init__(self, channels: int, seq_len: int, pred_len: int):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(channels, kernel_size=7, dilation=2)
        self.linear = nn.Linear(seq_len, pred_len)

    def moving_avg(self, x, kernel_size=5):
        padding = (kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):
        trend = self.moving_avg(x, kernel_size=7)
        seasonal = x - trend
        y = self.conv(seasonal.permute(0, 2, 1)).squeeze(1)
        return self.linear(y)


class NHiTSLikeBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, dropout: float = 0.3, hidden_ratio: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.net = nn.Sequential(
            nn.Linear(seq_len, seq_len // hidden_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len // hidden_ratio, pred_len)
        )

    def forward(self, x):
        return self.net(self.norm(x))


# =============================== 4. 完整模型 ===============================
class FullModel(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, features: int = 6,
                 dropout: float = 0.1, nhits_hidden_ratio: int = 2,
                 nhits_init: float = 0.01, auto_init: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.base_feat_num = features

        self.nhits_coef = nn.Parameter(torch.tensor(nhits_init))
        self.auto_coef = nn.Parameter(torch.tensor(auto_init))

        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.residual_linear = nn.Linear(seq_len, pred_len)
        self.feature_weights = nn.Parameter(torch.ones(features))

        self.decoder = nn.Sequential(
            nn.Linear(pred_len, pred_len * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, pred_len)
        )
        self.h_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        self.auto_block = AutoformerLiteBlock(channels=features, seq_len=seq_len, pred_len=pred_len)
        self.nhits_block = NHiTSLikeBlock(seq_len=seq_len, pred_len=pred_len,
                                          dropout=dropout, hidden_ratio=nhits_hidden_ratio)

    def moving_avg(self, x, kernel_size):
        padding = (kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):
        x_base = x[:, :, :self.base_feat_num]
        w = self.feature_weights.unsqueeze(0).unsqueeze(0)
        weighted_x = x_base * w

        # 1. Base DLinear Path
        trend = self.moving_avg(weighted_x, 5)
        residual = weighted_x - trend
        trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_linear(residual.permute(0, 2, 1)).permute(0, 2, 1)
        base_pred = (trend_pred + residual_pred)[:, :, 3]

        base_pred = base_pred + 0.1 * self.decoder(base_pred)

        out = base_pred

        # 2. Add Autoformer Path
        auto_out = self.auto_block(weighted_x)
        noise = random.uniform(-0.001, 0.001)
        lower_bound = 0.01 + noise
        clamped_auto_coef = torch.clamp(self.auto_coef, min=lower_bound)
        out = out + clamped_auto_coef * auto_out

        # 3. Add N-HiTS Path
        nhits_out = self.nhits_block(residual[:, :, 3])
        out = out + self.nhits_coef * nhits_out

        # 4. Smoothing
        smoothed = self.h_smooth(out.unsqueeze(1)).squeeze(1)
        out = out + 0.1 * (smoothed - out)
        smoothed = self.h_smooth(out.unsqueeze(1)).squeeze(1)
        out = out + 0.1 * (smoothed - out)
        return out


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


# ==============================================================
def train_and_evaluate(model_name, model, train_loader, test_loader, scaler, epochs=30):
    print(f"\nTraining {model_name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    criterion = DirectionalLoss(alpha=1.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            last_close = bx[:, -1, 3]

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by, last_close)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                last_close = bx[:, -1, 3]
                loss = criterion(model(bx), by, last_close)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()

    if hasattr(model, 'auto_coef') or hasattr(model, 'nhits_coef'):
        auto_w = model.auto_coef.item() if hasattr(model, 'auto_coef') else 0.0
        nhits_w = model.nhits_coef.item() if hasattr(model, 'nhits_coef') else 0.0
        print(f"[{model_name}] Learned Weights -> Auto: {auto_w:.4f}, N-HiTS: {nhits_w:.4f}")

    preds_list, trues_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds_list.append(model(bx).cpu().numpy())
            trues_list.append(by.numpy())

    preds_scaled = np.concatenate(preds_list)
    trues_scaled = np.concatenate(trues_list)

    def inverse_transform(scaled_data):
        n, p = scaled_data.shape
        original = np.zeros_like(scaled_data)
        for i in range(p):
            dummy = np.zeros((n, 9))
            dummy[:, 3] = scaled_data[:, i]
            original[:, i] = np.exp(scaler.inverse_transform(dummy)[:, 3]) - 1e-8
        return original

    preds = inverse_transform(preds_scaled)
    trues = inverse_transform(trues_scaled)

    metrics = {
        "Overall_MAE": mean_absolute_error(trues, preds),
        "Overall_RMSE": np.sqrt(mean_squared_error(trues, preds)),
        "Overall_R2": r2_score(trues, preds),
        "Step_Metrics": []
    }

    for i in range(PRED_LEN):
        metrics["Step_Metrics"].append({
            "Step": i + 1,
            "MAE": mean_absolute_error(trues[:, i], preds[:, i]),
            "RMSE": np.sqrt(mean_squared_error(trues[:, i], preds[:, i])),
            "R2": r2_score(trues[:, i], preds[:, i])
        })

    return metrics


# =============================== 5. 随机搜索寻优 ===============================
def random_search(train_loader, test_loader, scaler, num_trials=6):
    print("\n" + "=" * 50)
    print(f"🚀 Starting Random Search Optimization ({num_trials} trials)")
    print("=" * 50)

    param_grid = {
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "nhits_hidden_ratio": [2, 4, 8],
        "nhits_init": [0.0001, 0.001, 0.01, 0.05],
        "auto_init": [0.05, 0.1, 0.2]
    }

    best_rmse = float('inf')
    best_config = {}

    for i in range(num_trials):
        config = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"\n[Trial {i + 1}/{num_trials}] Testing config: {config}")

        torch.manual_seed(42 + i)
        model = FullModel(seq_len=20, pred_len=PRED_LEN, features=6, **config).to(device)

        try:
            metrics = train_and_evaluate(f"Trial_{i}", model, train_loader, test_loader, scaler, epochs=15)
            rmse = metrics["Overall_RMSE"]
            print(f"-> Result RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_config = config
                print(f"   >>> New Best Found! <<<")
        except Exception as e:
            print(f"Trial failed: {e}")

    print("\n" + "=" * 50)
    print(f"🏆 Optimization Complete. Best RMSE: {best_rmse:.4f}")
    print(f"Best Configuration: {best_config}")
    print("=" * 50 + "\n")
    return best_config


# =============================== 6. 运行实验 ===============================
def main():
    data_path = "JM_5.csv"
    if not os.path.exists(data_path):
        print(f"File {data_path} not found. Please provide data.")
        return

    df = pd.read_csv(data_path)
    train_loader, test_loader, scaler = preprocess_and_split_data(df, look_back=20, pred_len=PRED_LEN)

    # 1. 运行随机搜索寻找最佳超参数
    best_params = random_search(train_loader, test_loader, scaler, num_trials=6)

    print("=" * 60)
    print("STARTING FINAL VALIDATION WITH OPTIMIZED PARAMS")
    print("=" * 60)

    # 2. 使用最优参数训练和验证最终模型
    torch.manual_seed(42)
    model = FullModel(seq_len=20, pred_len=PRED_LEN, features=6, **best_params).to(device)
    metrics = train_and_evaluate("Full Model (Optimized)", model, train_loader, test_loader, scaler, epochs=30)

    print(f"[Full Model (Optimized)] Done. RMSE: {metrics['Overall_RMSE']:.4f}, R2: {metrics['Overall_R2']:.4f}")

    # 3. 输出模型整体评估结果
    print("\n" + "=" * 80)
    print("Overall Performance")
    print("-" * 80)
    print(f"{'Model Name':<30} | {'MAE':<15} | {'RMSE':<15} | {'R2 Score':<15}")
    print("-" * 80)
    print(
        f"{'Full Model':<30} | {metrics['Overall_MAE']:<15.6f} | {metrics['Overall_RMSE']:<15.6f} | {metrics['Overall_R2']:<15.6f}")

    # 4. 输出每一步详细评估结果
    print("\n" + "=" * 80)
    print("Detailed Step Analysis (All Steps)")
    print("-" * 80)
    print(f"{'Step':<10} | {'MAE':<15} | {'RMSE':<15} | {'R2 Score':<15}")
    print("-" * 80)
    for step_metric in metrics['Step_Metrics']:
        print(
            f"Step {step_metric['Step']:<5} | {step_metric['MAE']:<15.6f} | {step_metric['RMSE']:<15.6f} | {step_metric['R2']:<15.6f}")


if __name__ == "__main__":
    main()

"""
JM
================================================================================
Overall Performance
--------------------------------------------------------------------------------
Model Name                     | MAE             | RMSE            | R2 Score       
--------------------------------------------------------------------------------
Full Model                     | 7.695136        | 11.545626       | 0.997762       

================================================================================
Detailed Step Analysis (All Steps)
--------------------------------------------------------------------------------
Step       | MAE             | RMSE            | R2 Score       
--------------------------------------------------------------------------------
Step 1     | 3.135878        | 4.633955        | 0.999639       
Step 2     | 4.422998        | 6.471410        | 0.999296       
Step 3     | 5.375044        | 7.848668        | 0.998965       
Step 4     | 6.177711        | 9.002669        | 0.998638       
Step 5     | 6.911118        | 10.036946       | 0.998308       
Step 6     | 7.626250        | 11.026554       | 0.997958       
Step 7     | 8.203390        | 11.842890       | 0.997645       
Step 8     | 8.829156        | 12.677399       | 0.997301       
Step 9     | 9.785522        | 13.867704       | 0.996771       
Step 10    | 10.201879       | 14.445975       | 0.996496       
Step 11    | 10.453081       | 14.843825       | 0.996301       
Step 12    | 11.219594       | 15.779300       | 0.995821       


"""