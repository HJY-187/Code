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

# =============================== 1. Global Settings ===============================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LEN = 12


# =============================== 2. Data Processing (Keep original target logic) ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        # Input features: first 6 columns (open/high/low/close/volume/open_interest)
        # Although the original data has 9 columns, only the first 6 are used as Base in forward;
        # the last 3 columns exist in input only as auxiliary information
        x = self.data[idx:idx + self.seq_len, :]
        # Prediction target: future pred_len values of close column (index 3)
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, 3]
        # Current price (used for Directional Loss)
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

    # Technical indicator calculation (keep original logic)
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

    # 9 features
    features = df_work[[
        "open", "high", "low", "close", "volume", "open_interest",
        "ATR", "RSI", "roll_return"
    ]].values

    dates_work = df_work["datetime"].values

    # Split (8:2)
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


def denormalize_predictions(scaled_preds, scaler):
    """Denormalization utility"""
    n, p = scaled_preds.shape
    original = np.zeros_like(scaled_preds)
    for i in range(p):
        dummy = np.zeros((n, 9))
        dummy[:, 3] = scaled_preds[:, i]
        den = scaler.inverse_transform(dummy)
        original[:, i] = np.exp(den[:, 3]) - 1e-8
    return original


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
    Model class with ablation capability
    Default parameters remain consistent with the original model,
    but components can be controlled via arguments
    """

    def __init__(self, seq_len: int, pred_len: int, features: int = 6,
                 kernel_size: int = 5,
                 use_nhits: bool = True, use_auto: bool = True,
                 dropout: float = 0.3, nhits_hidden_ratio: int = 4,
                 nhits_init: float = 0.001, auto_init: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.base_feat_num = features  # Here it is 6, corresponding to the first 6 columns
        self.kernel_size = kernel_size
        self.use_nhits = use_nhits
        self.use_auto = use_auto

        # Learnable Fusion Weights
        self.nhits_coef = nn.Parameter(torch.tensor(nhits_init))
        self.auto_coef = nn.Parameter(torch.tensor(auto_init))

        # DLinear Base
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.residual_linear = nn.Linear(seq_len, pred_len)
        self.feature_weights = nn.Parameter(torch.ones(features))
        self.decoder = nn.Sequential(
            nn.Linear(pred_len, pred_len * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, pred_len)
        )

        # Optional Blocks
        if self.use_auto:
            # AS1 block needs full features dim if passed, but here features=6
            self.auto_block = AS1(channels=features, seq_len=seq_len, pred_len=pred_len)

        if self.use_nhits:
            self.nhits_block = AS2(seq_len=seq_len, pred_len=pred_len,
                                              dropout=dropout, hidden_ratio=nhits_hidden_ratio)

        # Smoothing
        self.h_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def moving_avg(self, x, kernel_size):
        padding = (kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):
        # x shape: [B, S, 9] -> Use first 6 for main model logic
        x_base = x[:, :, :self.base_feat_num]

        w = self.feature_weights.unsqueeze(0).unsqueeze(0)
        weighted_x = x_base * w

        # 1. Base DLinear
        trend = self.moving_avg(weighted_x, self.kernel_size)
        residual = weighted_x - trend
        trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_linear(residual.permute(0, 2, 1)).permute(0, 2, 1)
        base_pred = (trend_pred + residual_pred)[:, :, 3]  # Close index is 3
        base_pred = base_pred + 0.1 * self.decoder(base_pred)

        out = base_pred

        # 2. AS1 Path
        if self.use_auto:
            auto_out = self.auto_block(weighted_x)
            out = out + self.auto_coef * auto_out

        # 3. AS2 Path (Modeling Residuals of Close Price)
        if self.use_nhits:
            nhits_out = self.nhits_block(residual[:, :, 3])
            out = out + self.nhits_coef * nhits_out

        # 4. Smoothing
        smoothed = self.h_smooth(out.unsqueeze(1)).squeeze(1)
        out = out + 0.1 * (smoothed - out)
        smoothed = self.h_smooth(out.unsqueeze(1)).squeeze(1)  # Double smoothing as per original
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
        # Penalty if signs differ
        penalty = F.relu(-1.0 * true_diff * pred_diff)
        return num_loss + self.alpha * torch.mean(penalty)


# =============================== 4. Training & Evaluation Utilities (Fusion Version) ===============================
def train_and_evaluate(model_name, model, train_loader, test_loader, scaler, epochs=30, verbose=True):
    """
    Generic training function supporting ablation experiments and random search.
    After training, it automatically performs full prediction and computes detailed metrics.
    """
    if verbose:
        print(f"\nTraining {model_name}...")

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = DirectionalLoss(alpha=1.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    early_stop_patience = 12

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by, last_close_tensor in train_loader:
            bx, by, last_close_tensor = bx.to(device), by.to(device), last_close_tensor.to(device)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by, last_close_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by, last_close_tensor in test_loader:
                bx, by, last_close_tensor = bx.to(device), by.to(device), last_close_tensor.to(device)
                pred = model(bx)
                loss = criterion(pred, by, last_close_tensor)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose: print(f"  Early stop at epoch {epoch + 1}")
                break

    # --- Load Best Weights ---
    model.load_state_dict(best_state)
    model.eval()

    # --- Print learned coefficients ---
    if verbose and (hasattr(model, 'auto_coef') or hasattr(model, 'nhits_coef')):
        info = f"[{model_name}] Learned Weights -> "
        if model.use_auto: info += f"Auto: {model.auto_coef.item():.4f} "
        if model.use_nhits: info += f"AS2: {model.nhits_coef.item():.4f}"
        print(info)

    # --- Final Evaluation & Denormalization ---
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx = bx.to(device)
            preds_scaled.append(model(bx).cpu().numpy())
            trues_scaled.append(by.numpy())

    preds_scaled = np.concatenate(preds_scaled)
    trues_scaled = np.concatenate(trues_scaled)

    # Use RobustScaler for inverse transform
    preds = denormalize_predictions(preds_scaled, scaler)
    trues = denormalize_predictions(trues_scaled, scaler)

    # --- Compute Metrics ---
    metrics = {
        "Overall_MAE": mean_absolute_error(trues, preds),
        "Overall_RMSE": np.sqrt(mean_squared_error(trues, preds)),
        "Overall_R2": r2_score(trues, preds),
        "Step_Metrics": [],
        "preds": preds,  # Save predictions for later file output
        "trues": trues
    }

    # Compute per-step metrics
    for i in range(PRED_LEN):
        metrics["Step_Metrics"].append({
            "Step": i + 1,
            "RMSE": np.sqrt(mean_squared_error(trues[:, i], preds[:, i])),
            "R2": r2_score(trues[:, i], preds[:, i])
        })

    return metrics, model


# =============================== 5. Random Search Optimization ===============================
def random_search(train_loader, test_loader, scaler, num_trials=6):
    print("\n" + "=" * 50)
    print(f"🚀 Starting Random Search Optimization ({num_trials} trials)")
    print("=" * 50)

    # Define search space
    param_grid = {
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "nhits_hidden_ratio": [2, 4, 8],  # 2=Large capacity, 8=Small capacity
        "nhits_init": [0.0001, 0.001, 0.01],
        "auto_init": [0.05, 0.1, 0.2]
    }

    best_rmse = float('inf')
    best_config = {
        "dropout": 0.3, "nhits_hidden_ratio": 4, "nhits_init": 0.001, "auto_init": 0.1
    }  # Default fallback

    for i in range(num_trials):
        # Random sampling
        config = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"\n[Trial {i + 1}/{num_trials}] Testing config: {config}")

        # Use full model configuration for search
        torch.manual_seed(42 + i)
        model = DENet(
            seq_len=20, pred_len=PRED_LEN, features=6,
            use_nhits=True, use_auto=True,
            **config
        ).to(device)

        # Quick training (15 epochs) to assess potential
        try:
            metrics, _ = train_and_evaluate(f"Trial_{i}", model, train_loader, test_loader, scaler, epochs=15,
                                            verbose=False)
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


# =============================== 6. Main Program ===============================
def main():
    data_path = "I_5.csv"
    if not os.path.exists(data_path):
        print(f"File {data_path} not found. Please provide data.")
        return

    # 1. Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Automatically determine look_back
    look_back = 48 if "30" in data_path else 20

    train_loader, test_loader, scaler, dates_work, train_size, look_back = preprocess_and_split_data(
        df, look_back=look_back, pred_len=PRED_LEN
    )

    # 2. Run random search optimization
    best_params = random_search(train_loader, test_loader, scaler, num_trials=5)

    # 3. Define ablation experiment groups
    experiments = {
        "Baseline (base)": {
            "use_nhits": False, "use_auto": False,
            **best_params
        },
        "Model A (+AS1)": {
            "use_nhits": False, "use_auto": True,
            **best_params
        },
        "Model B (+AS2 Optimized)": {
            "use_nhits": True, "use_auto": False,
            **best_params
        },
        "Full Model (Optimized)": {
            "use_nhits": True, "use_auto": True,
            **best_params
        },
    }

    results = {}
    final_best_model = None
    final_metrics = None

    print("=" * 60)
    print("STARTING FINAL ABLATION STUDY WITH OPTIMIZED PARAMS")
    print("=" * 60)

    # 4. Execute ablation experiments
    for name, config in experiments.items():
        torch.manual_seed(42)  # Strict seed control to ensure fair comparison

        # Initialize model
        model = DENet(
            seq_len=look_back, pred_len=PRED_LEN, features=6,
            **config
        ).to(device)

        # Full training (30 epochs; early stopping may shorten it. Original code used 160; here set to 30 for demo)
        metrics, trained_model = train_and_evaluate(name, model, train_loader, test_loader, scaler, epochs=50)
        results[name] = metrics

        print(f"[{name}] Done. RMSE: {metrics['Overall_RMSE']:.4f}, R2: {metrics['Overall_R2']:.4f}")

        # If full model, save for final CSV generation
        if name == "Full Model (Optimized)":
            final_best_model = trained_model
            final_metrics = metrics

    # 5. Output comparison table
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

    # 6. Generate final prediction file (keep original functionality)
    if final_best_model is not None:
        print("\n" + "=" * 60)
        print("GENERATING FINAL PREDICTION CSV")
        print("=" * 60)

        preds = final_metrics['preds']
        # Align dates
        res_len = preds.shape[0]
        start_idx = train_size + 2 * look_back
        res_dates = dates_work[start_idx: start_idx + res_len]

        df_out = pd.DataFrame()
        df_out['datetime'] = res_dates
        for k in range(PRED_LEN):
            df_out[f'pred_step_{k + 1}'] = preds[:, k]
        df_out['pred_avg_1hr'] = preds.mean(axis=1)

        save_path = os.path.join("..", "Strategy", "pred_minute_advanced_opt.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_out.to_csv(save_path, index=False, encoding='utf-8')
        print(f"Prediction results saved to: {save_path}")

        # Save best model
        torch.save(final_best_model.state_dict(), "best_opt_model.pth")
        print("Best model weights saved to: best_opt_model.pth")


if __name__ == "__main__":
    main()

"""
D:\miniconda\envs\Py123\python.exe "D:\桌面\Deeplearning\Dlinear\I(1)\分钟消融\消融2.py" 
Loading data from I_5.csv...

==================================================
Starting Random Search Optimization (5 trials)
==================================================

[Trial 1/5] Testing config: {'dropout': 0.2, 'nhits_hidden_ratio': 2, 'nhits_init': 0.01, 'auto_init': 0.1}
-> Result RMSE: 5.2805
   >>> New Best Found! <<<

[Trial 2/5] Testing config: {'dropout': 0.3, 'nhits_hidden_ratio': 2, 'nhits_init': 0.0001, 'auto_init': 0.2}
-> Result RMSE: 5.0246
   >>> New Best Found! <<<

[Trial 3/5] Testing config: {'dropout': 0.2, 'nhits_hidden_ratio': 8, 'nhits_init': 0.01, 'auto_init': 0.2}
-> Result RMSE: 4.9909
   >>> New Best Found! <<<

[Trial 4/5] Testing config: {'dropout': 0.2, 'nhits_hidden_ratio': 8, 'nhits_init': 0.001, 'auto_init': 0.05}
-> Result RMSE: 5.0047

[Trial 5/5] Testing config: {'dropout': 0.2, 'nhits_hidden_ratio': 2, 'nhits_init': 0.0001, 'auto_init': 0.05}
-> Result RMSE: 5.0536

==================================================
Optimization Complete. Best RMSE: 4.9909
Best Configuration: {'dropout': 0.2, 'nhits_hidden_ratio': 8, 'nhits_init': 0.01, 'auto_init': 0.2}
==================================================

============================================================
STARTING FINAL ABLATION STUDY WITH OPTIMIZED PARAMS
============================================================

Training Baseline (base)...
[Baseline (base)] Learned Weights -> 
[Baseline (base)] Done. RMSE: 4.9086, R2: 0.9847

Training Model A (+AS1)...
[Model A (+AS1)] Learned Weights -> Auto: 0.5131 
[Model A (+AS1)] Done. RMSE: 4.8355, R2: 0.9851

Training Model B (+AS2 Optimized)...
[Model B (+AS2 Optimized)] Learned Weights -> AS2: -0.0006
[Model B (+AS2 Optimized)] Done. RMSE: 4.8954, R2: 0.9848

Training Full Model (Optimized)...
[Full Model (Optimized)] Learned Weights -> Auto: 0.5516 AS2: -0.0000
[Full Model (Optimized)] Done. RMSE: 4.8321, R2: 0.9852

================================================================================
Model Name                     | MAE        | RMSE       | R2 Score  
--------------------------------------------------------------------------------
Baseline (base)        | 3.314626 | 4.908586 | 0.984697
Model A (+AS1)          | 3.217394 | 4.835517 | 0.985149
Model B (+AS2 Optimized)    | 3.288767 | 4.895439 | 0.984779
Full Model (Optimized)         | 3.209262 | 4.832119 | 0.985170

================================================================================
Detailed Step Analysis (Step 1, 6, 12 RMSE)
--------------------------------------------------------------------------------
Model Name                     | Step 1     | Step 6     | Step 12   
Baseline (base)        | 2.396089 | 4.881921 | 6.472212
Model A (+AS1)          | 2.088563 | 4.660959 | 6.456911
Model B (+AS2 Optimized)    | 2.234645 | 4.706626 | 6.501599
Full Model (Optimized)         | 2.039120 | 4.660242 | 6.455535

============================================================
GENERATING FINAL PREDICTION CSV
============================================================
Prediction results saved to: ..\Strategy\pred_minute_advanced_opt.csv
Best model weights saved to: best_opt_model.pth

Process finished with exit code 0

"""
