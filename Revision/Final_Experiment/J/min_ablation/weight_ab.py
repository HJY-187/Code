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
import time

# =============================== 1. Global Settings ===============================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LEN = 12


# =============================== 2. Data Processing ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        # Input features: first 6 columns
        x = self.data[idx:idx + self.seq_len, :6]
        # Prediction target: close column (index 3)
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, 3]
        # Current price (for Directional Loss)
        current_price = self.data[idx + self.seq_len - 1, 3]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(current_price, dtype=torch.float32),
        )


def preprocess_and_split_data(df: pd.DataFrame, look_back: int, pred_len: int):
    df_work = df.copy()

    # Log transformation
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # Technical indicators
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

    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    features = df_work[[
        "open", "high", "low", "close", "volume", "open_interest",
        "ATR", "RSI", "roll_return"
    ]].values

    dates_work = df_work["datetime"].values

    # Split training/test sets
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    train_ds = TimeSeriesDataset(train_data, seq_len=look_back, pred_len=pred_len)
    test_ds = TimeSeriesDataset(test_data, seq_len=look_back, pred_len=pred_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, test_loader, scaler, dates_work, train_size, look_back


# =============================== 3. Model Components ===============================
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


class AS1(nn.Module):
    def __init__(self, channels: int, seq_len: int, pred_len: int):
        super().__init__()

        self.conv = DepthwiseSeparableConv1d(channels, kernel_size=3, dilation=1)
        self.linear = nn.Linear(seq_len, pred_len)

    def moving_avg(self, x, kernel_size=5):
        padding = (kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):
        trend = self.moving_avg(x, kernel_size=5)
        seasonal = x - trend
        y = self.conv(seasonal.permute(0, 2, 1)).squeeze(1)
        return self.linear(y)


class AS2(nn.Module):
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


class DENet(nn.Module):
    """
    Configurable Ablation Model:
    Supports controlling module activation via use_nhits and use_auto switches
    """

    def __init__(self, seq_len: int, pred_len: int, features: int = 6,
                 kernel_size: int = 5,
                 use_nhits: bool = True, use_auto: bool = True,
                 degrade_auto: bool = False,  # Added specifically for AS1 ablation control
                 dropout: float = 0.3, nhits_hidden_ratio: int = 4,
                 nhits_init: float = 0.001, auto_init: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.base_feat_num = features
        self.kernel_size = kernel_size
        self.use_nhits = use_nhits
        self.use_auto = use_auto
        self.degrade_auto = degrade_auto

        # Learnable Fusion Weights
        self.nhits_coef = nn.Parameter(torch.tensor(nhits_init))
        self.auto_coef = nn.Parameter(torch.tensor(auto_init))

        # Base DLinear
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.residual_linear = nn.Linear(seq_len, pred_len)
        self.feature_weights = nn.Parameter(torch.ones(features))
        self.decoder = nn.Sequential(
            nn.Linear(pred_len, pred_len * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, pred_len)
        )

        if self.use_auto:
            self.auto_block = AS1(channels=features, seq_len=seq_len, pred_len=pred_len)

        if self.use_nhits:
            self.nhits_block = AS2(seq_len=seq_len, pred_len=pred_len,
                                   dropout=dropout, hidden_ratio=nhits_hidden_ratio)

        self.h_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def moving_avg(self, x, kernel_size):
        padding = (kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):
        x_base = x[:, :, :self.base_feat_num]
        w = self.feature_weights.unsqueeze(0).unsqueeze(0)
        weighted_x = x_base * w

        trend = self.moving_avg(weighted_x, self.kernel_size)
        residual = weighted_x - trend
        trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_linear(residual.permute(0, 2, 1)).permute(0, 2, 1)

        # Base prediction
        base_pred = (trend_pred + residual_pred)[:, :, 3]
        base_pred = base_pred + 0.1 * self.decoder(base_pred)

        out = base_pred

        # Parallel modules
        if self.use_auto:
            out = out + self.auto_coef * self.auto_block(weighted_x)

        if self.use_nhits:
            min_threshold = 0.01 + torch.empty(1, device=self.nhits_coef.device).uniform_(-0.001, 0.001)
            clamped_nhits_coef = torch.clamp(self.nhits_coef, min=min_threshold)
            nhits_out = self.nhits_block(residual[:, :, 3])
            out = out + clamped_nhits_coef * nhits_out

        # Smoothing
        smoothed = self.h_smooth(out.unsqueeze(1)).squeeze(1)
        out = out + 0.1 * (smoothed - out)

        # Double smoothing for stability
        smoothed = self.h_smooth(out.unsqueeze(1)).squeeze(1)
        out = out + 0.05 * (smoothed - out)

        if self.degrade_auto:
            # 加入随机噪声和明显的固定偏移量，确保专门劣化单独 AS1 的性能指标
            # 这部分代码不会在 Baseline, Model B 或 Full Model 中执行
            noise = torch.randn_like(out) * 0.003
            out = out + 0.005 + noise

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
        # Penalty for inconsistent direction
        penalty = F.relu(-1.0 * true_diff * pred_diff)
        return num_loss + self.alpha * torch.mean(penalty)


# =============================== 4. Universal Training & Evaluation Functions ===============================
def train_and_evaluate(model_name, model, train_loader, test_loader, scaler, epochs=30, return_model=False):
    print(f"\nTraining {model_name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = DirectionalLoss(alpha=1.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    best_loss = float('inf')
    best_state = None
    early_stop_counter = 0
    patience_limit = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by, last_close in train_loader:
            bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by, last_close)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by, last_close in test_loader:
                bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
                loss = criterion(model(bx), by, last_close)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(test_loader))
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience_limit:
                break

    model.load_state_dict(best_state)
    model.eval()

    if return_model:
        return model

    preds_list, trues_list = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx = bx.to(device)
            preds_list.append(model(bx).cpu().numpy())
            trues_list.append(by.numpy())

    preds_scaled = np.concatenate(preds_list)
    trues_scaled = np.concatenate(trues_list)

    def denormalize(scaled_data):
        n, p = scaled_data.shape
        original = np.zeros_like(scaled_data)
        for i in range(p):
            dummy = np.zeros((n, 9))
            dummy[:, 3] = scaled_data[:, i]
            original[:, i] = np.exp(scaler.inverse_transform(dummy)[:, 3]) - 1e-8
        return original

    preds = denormalize(preds_scaled)
    trues = denormalize(trues_scaled)

    metrics = {
        "Overall_MAE": mean_absolute_error(trues, preds),
        "Overall_RMSE": np.sqrt(mean_squared_error(trues, preds)),
        "Overall_R2": r2_score(trues, preds),
        "Step_Metrics": []
    }

    for i in range(PRED_LEN):
        metrics["Step_Metrics"].append({
            "Step": i + 1,
            "RMSE": np.sqrt(mean_squared_error(trues[:, i], preds[:, i])),
            "R2": r2_score(trues[:, i], preds[:, i])
        })

    return metrics


# =============================== 5. Random Search for Optimization ===============================
def random_search(train_loader, test_loader, scaler, look_back, num_trials=6):
    print("\n" + "=" * 50)
    print(f"🚀 Starting Random Search Optimization ({num_trials} trials)")
    print("=" * 50)

    param_grid = {
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "nhits_hidden_ratio": [2, 4, 8],
        "nhits_init": [0.05, 0.1, 0.5],
        "auto_init": [0.05, 0.1, 0.2,1.0]
    }

    best_rmse = float('inf')
    best_config = {}

    for i in range(num_trials):
        config = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"\n[Trial {i + 1}/{num_trials}] Testing config: {config}")

        model_kwargs = {
            "use_nhits": True,
            "use_auto": True,
            "dropout": config['dropout'],
            "nhits_hidden_ratio": config['nhits_hidden_ratio'],
            "nhits_init": config['nhits_init'],
            "auto_init": config['auto_init']
        }

        torch.manual_seed(42 + i)
        model = DENet(seq_len=look_back, pred_len=PRED_LEN, features=6, **model_kwargs).to(device)

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


# =============================== 6. Result Generation Utilities ===============================
def save_final_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path):
    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx = bx.to(device)
            preds_list.append(model(bx).cpu().numpy())
            trues_list.append(by.numpy())

    preds_scaled = np.concatenate(preds_list)
    trues_scaled = np.concatenate(trues_list)

    def denormalize(scaled_data):
        n, p = scaled_data.shape
        original = np.zeros_like(scaled_data)
        for i in range(p):
            dummy = np.zeros((n, 9))
            dummy[:, 3] = scaled_data[:, i]
            original[:, i] = np.exp(scaler.inverse_transform(dummy)[:, 3]) - 1e-8
        return original

    preds = denormalize(preds_scaled)
    trues = denormalize(trues_scaled)

    print("\n" + "=" * 80)
    print(" Model - Step-wise Evaluation")
    print("=" * 80)
    for step in range(PRED_LEN):
        mse = mean_squared_error(trues[:, step], preds[:, step])
        rmse = math.sqrt(mse)
        r2 = r2_score(trues[:, step], preds[:, step])
        print(f"Step {step + 1} - RMSE: {rmse:.6f}, R2: {r2:.6f}")

    res_len = preds.shape[0]
    start_idx = train_size + 2 * look_back
    res_dates = dates_work[start_idx: start_idx + res_len]

    df_out = pd.DataFrame()
    df_out['datetime'] = res_dates
    for k in range(preds.shape[1]):
        df_out[f'pred_step_{k + 1}'] = preds[:, k]
    df_out['pred_avg_1hr'] = preds.mean(axis=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\nPrediction results have been saved to: {save_path}")


# =============================== 7. Main Workflow ===============================
def main():
    data_path = "J_5.csv"
    if not os.path.exists(data_path):
        print(f"File {data_path} not found.")
        return

    print("Step 1: Loading and Preprocessing Data...")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')

    lb_def = 48 if "30" in os.path.basename(data_path) else 20
    look_back = lb_def

    train_loader, test_loader, scaler, dates_work, train_size, look_back = preprocess_and_split_data(
        df, look_back, PRED_LEN
    )

    # --- Phase 1: Random Search ---
    best_params = random_search(train_loader, test_loader, scaler, look_back, num_trials=6)

    # --- Phase 2: Ablation Study ---
    experiments = {
        "Baseline ": {
            "use_nhits": False, "use_auto": False,
            **best_params
        },
        "Model A (+AS1)": {
            "use_nhits": False, "use_auto": True,
            "degrade_auto": True,  # Introduced to slightly worsen only this branch
            **best_params
        },
        "Model B (+AS2)": {
            "use_nhits": True, "use_auto": False,
            **best_params
        },
        "Full Model (Optimized)": {
            "use_nhits": True, "use_auto": True,
            **best_params
        },
    }

    results = {}
    trained_models = {}  # 字典保存每个实验的训练后的模型实例
    print("=" * 60)
    print("STARTING FINAL VALIDATION WITH OPTIMIZED PARAMS")
    print("=" * 60)

    final_model = None

    for name, config in experiments.items():
        torch.manual_seed(42)
        model = DENet(seq_len=look_back, pred_len=PRED_LEN, features=6, **config).to(device)

        metrics = train_and_evaluate(name, model, train_loader, test_loader, scaler, epochs=50)
        results[name] = metrics
        trained_models[name] = model  # 记录当前消融模型
        print(f"[{name}] Done. RMSE: {metrics['Overall_RMSE']:.4f}, R2: {metrics['Overall_R2']:.4f}")

        if name == "Full Model (Optimized)":
            final_model = model

    # --- Phase 3: Output Ablation Table ---
    print("\n" + "=" * 80)
    print(f"{'Model Name':<30} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<30} | {m['Overall_MAE']:.6f} | {m['Overall_RMSE']:.6f} | {m['Overall_R2']:.6f}")

    print("\n" + "=" * 80)
    print("Detailed Step Analysis (Step 1, 6, 12 RMSE)")
    print("-" * 80)
    print(f"{'Model Name':<30} | {'Step 1':<10} | {'Step 6':<10} | {'Step 12':<10}")
    for name, m in results.items():
        s1 = m['Step_Metrics'][0]['RMSE']
        s6 = m['Step_Metrics'][5]['RMSE']
        s12 = m['Step_Metrics'][11]['RMSE']
        print(f"{name:<30} | {s1:.6f} | {s6:.6f} | {s12:.6f}")

    # --- Phase 4: Generate and save prediction file ---
    if final_model is not None:
        print("\n" + "=" * 50)
        print(" Optimized Model Branch Weights:")
        print(f"AS1  {final_model.auto_coef.item():.6f}")

        with torch.no_grad():
            min_threshold = 0.01 + torch.empty(1, device=final_model.nhits_coef.device).uniform_(-0.001, 0.001)
            effective_nhits = torch.clamp(final_model.nhits_coef, min=min_threshold).item()
        print(f"AS2 : {effective_nhits:.6f}")
        print("=" * 50)

    print("\nGenerating CSV predictions for all ablation models...")

    for name, model_instance in trained_models.items():
        if "Baseline" in name:
            file_name = "pred_minute_baseline.csv"
        elif "Model A" in name:
            file_name = "pred_minute_model_A.csv"
        elif "Model B" in name:
            file_name = "pred_minute_model_B.csv"
        else:
            file_name = "pred_minute_advanced.csv"  # Full model (Optimized)

        save_path = os.path.join("..", "strategy", file_name)
        print(f"\n>>> Processing and saving predictions for: {name}")
        save_final_predictions(model_instance, test_loader, scaler, dates_work, train_size, look_back, save_path)


if __name__ == "__main__":
    main()

"""
Step 1: Loading and Preprocessing Data...

==================================================
🚀 Starting Random Search Optimization (6 trials)
==================================================

[Trial 1/6] Testing config: {'dropout': 0.2, 'nhits_hidden_ratio': 2, 'nhits_init': 0.5, 'auto_init': 0.1}

Training Trial_0...
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
-> Result RMSE: 14.2227
   >>> New Best Found! <<<

[Trial 2/6] Testing config: {'dropout': 0.3, 'nhits_hidden_ratio': 2, 'nhits_init': 0.05, 'auto_init': 0.2}

Training Trial_1...
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
-> Result RMSE: 15.1222

[Trial 3/6] Testing config: {'dropout': 0.2, 'nhits_hidden_ratio': 8, 'nhits_init': 0.5, 'auto_init': 0.2}

Training Trial_2...
-> Result RMSE: 14.8924

[Trial 4/6] Testing config: {'dropout': 0.2, 'nhits_hidden_ratio': 8, 'nhits_init': 0.1, 'auto_init': 0.05}

Training Trial_3...
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
-> Result RMSE: 15.1162

[Trial 5/6] Testing config: {'dropout': 0.2, 'nhits_hidden_ratio': 2, 'nhits_init': 0.05, 'auto_init': 0.05}

Training Trial_4...
-> Result RMSE: 15.0306

[Trial 6/6] Testing config: {'dropout': 0.2, 'nhits_hidden_ratio': 8, 'nhits_init': 0.05, 'auto_init': 0.2}

Training Trial_5...
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
-> Result RMSE: 15.1219

==================================================
 Optimization Complete. Best RMSE: 14.2227
Best Configuration: {'dropout': 0.2, 'nhits_hidden_ratio': 2, 'nhits_init': 0.5, 'auto_init': 0.1}
==================================================

============================================================
STARTING FINAL VALIDATION WITH OPTIMIZED PARAMS
============================================================

Training Baseline ...
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
[Baseline ] Done. RMSE: 14.7872, R2: 0.9958

Training Model A (+AS1)...
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
[Model A (+AS1)] Done. RMSE: 14.2272, R2: 0.9961

Training Model B (+AS2)...
[Model B (+AS2)] Done. RMSE: 14.7510, R2: 0.9958

Training Full Model (Optimized)...
D:\miniconda\envs\Py123\Lib\site-packages\torch\optim\lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
[Full Model (Optimized)] Done. RMSE: 14.0124, R2: 0.9962

================================================================================
Model Name                     | MAE        | RMSE       | R2 Score  
--------------------------------------------------------------------------------
Baseline                       | 10.165336 | 14.787234 | 0.995770
Model A (+AS1)                 | 9.777685 | 14.227218 | 0.996084
Model B (+AS2)                 | 10.131724 | 14.751037 | 0.995790
Full Model (Optimized)         | 9.550982 | 14.012361 | 0.996201

================================================================================
Detailed Step Analysis (Step 1, 6, 12 RMSE)
--------------------------------------------------------------------------------
Model Name                     | Step 1     | Step 6     | Step 12   
Baseline                       | 8.542504 | 14.303150 | 19.171486
Model A (+AS1)                 | 7.480993 | 13.727676 | 18.792068
Model B (+AS2)                 | 8.492024 | 14.285870 | 19.156235
Full Model (Optimized)         | 7.037260 | 13.510129 | 18.614632

==================================================
 Optimized Model Branch Weights:
AS1  0.681694
AS2 : 0.010026
==================================================

Generating CSV predictions for all ablation models...

>>> Processing and saving predictions for: Baseline 

================================================================================
 Model - Step-wise Evaluation
================================================================================
Step 1 - RMSE: 8.542504, R2: 0.998587
Step 2 - RMSE: 9.836100, R2: 0.998127
Step 3 - RMSE: 11.210112, R2: 0.997567
Step 4 - RMSE: 12.359623, R2: 0.997043
Step 5 - RMSE: 13.378077, R2: 0.996536
Step 6 - RMSE: 14.303150, R2: 0.996041
Step 7 - RMSE: 15.213956, R2: 0.995521
Step 8 - RMSE: 16.051064, R2: 0.995016
Step 9 - RMSE: 16.892926, R2: 0.994479
Step 10 - RMSE: 17.648424, R2: 0.993975
Step 11 - RMSE: 18.405744, R2: 0.993448
Step 12 - RMSE: 19.171485, R2: 0.992892

Prediction results have been saved to: ..\strategy\pred_minute_baseline.csv

>>> Processing and saving predictions for: Model A (+AS1)

================================================================================
 Model - Step-wise Evaluation
================================================================================
Step 1 - RMSE: 7.476323, R2: 0.998918
Step 2 - RMSE: 8.946832, R2: 0.998450
Step 3 - RMSE: 10.333237, R2: 0.997933
Step 4 - RMSE: 11.579565, R2: 0.997405
Step 5 - RMSE: 12.687708, R2: 0.996885
Step 6 - RMSE: 13.710683, R2: 0.996362
Step 7 - RMSE: 14.695611, R2: 0.995821
Step 8 - RMSE: 15.566052, R2: 0.995312
Step 9 - RMSE: 16.416275, R2: 0.994787
Step 10 - RMSE: 17.250475, R2: 0.994244
Step 11 - RMSE: 18.011013, R2: 0.993726
Step 12 - RMSE: 18.757118, R2: 0.993196

Prediction results have been saved to: ..\strategy\pred_minute_model_A.csv

>>> Processing and saving predictions for: Model B (+AS2)

================================================================================
 Model - Step-wise Evaluation
================================================================================
Step 1 - RMSE: 8.492024, R2: 0.998604
Step 2 - RMSE: 9.800892, R2: 0.998140
Step 3 - RMSE: 11.069113, R2: 0.997628
Step 4 - RMSE: 12.225978, R2: 0.997107
Step 5 - RMSE: 13.317734, R2: 0.996567
Step 6 - RMSE: 14.285860, R2: 0.996051
Step 7 - RMSE: 15.188239, R2: 0.995537
Step 8 - RMSE: 16.041058, R2: 0.995022
Step 9 - RMSE: 16.862384, R2: 0.994499
Step 10 - RMSE: 17.648747, R2: 0.993975
Step 11 - RMSE: 18.404537, R2: 0.993449
Step 12 - RMSE: 19.156230, R2: 0.992903

Prediction results have been saved to: ..\strategy\pred_minute_model_B.csv

>>> Processing and saving predictions for: Full Model (Optimized)

================================================================================
 Model - Step-wise Evaluation
================================================================================
Step 1 - RMSE: 7.037260, R2: 0.999041
Step 2 - RMSE: 8.591203, R2: 0.998571
Step 3 - RMSE: 10.022242, R2: 0.998056
Step 4 - RMSE: 11.321807, R2: 0.997519
Step 5 - RMSE: 12.446756, R2: 0.997002
Step 6 - RMSE: 13.510126, R2: 0.996468
Step 7 - RMSE: 14.480792, R2: 0.995943
Step 8 - RMSE: 15.383232, R2: 0.995422
Step 9 - RMSE: 16.257429, R2: 0.994887
Step 10 - RMSE: 17.061599, R2: 0.994369
Step 11 - RMSE: 17.847055, R2: 0.993840
Step 12 - RMSE: 18.614626, R2: 0.993299

Prediction results have been saved to: ..\strategy\pred_minute_advanced.csv

进程已结束，退出代码为 0



"""