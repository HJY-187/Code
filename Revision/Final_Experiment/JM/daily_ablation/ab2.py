import os
import math
import copy
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Seed
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================
LOOK_BACK = 22
PRED_LEN = 1
EPOCHS = 180
LR = 5e-4
BATCH_SIZE = 64
DATA_FILE = "JM_daily.csv"
OUTPUT_FILE = "../daily_ablation/pred_daily_advanced.csv"

# ----------------------------------------------------------------
# Ablation settings
# "learnable_only" -> only current DENet
# "fixed_only"     -> only run all predefined fixed (beta, gamma) combinations
# "all"            -> run learnable + all predefined fixed combinations
ABLATION_MODE = "all"

# ----------------------------------------------------------------
# Predefined fixed fusion weights
# IMPORTANT:
# These are NOT used to select the best pair afterwards.
# They are treated as independent fixed baselines and reported one by one.
# Fixed(1,1) is included as one case in the grid.
PREDEFINED_FIXED_PAIRS = [
    (1.0, 1.0),  # original manual fixed baseline, now treated as one grid case
    (0.0, 0.0),
    (0.01, 0.01),
    (0.05, 0.05),
    (0.1, 0.01),
    (0.1, 0.05),
    (0.1, 0.1),
    (0.2, 0.01),
    (0.2, 0.05),
    (0.2, 0.1),
    (0.2, 0.2),
    (0.5, 0.1),
]

# ----------------------------------------------------------------
# Multi-asset / multi-frequency experiment list
EXPERIMENTS = [
    {
        "asset_name": "JM",
        "frequency": "daily",
        "data_file": DATA_FILE,
        "output_file": OUTPUT_FILE
    },
]


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - PRED_LEN + 1

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.seq_len, :]
        y = self.data[idx + self.seq_len, 3]
        previous_close = self.data[idx + self.seq_len - 1, 3]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
            torch.tensor(previous_close, dtype=torch.float32)
        )


def preprocess_data(df: pd.DataFrame, look_back: int):
    df_work = df.copy()

    # 1. log()
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # 2. calendar features
    dt = df_work['datetime']
    df_work['day_of_week'] = dt.dt.dayofweek
    df_work['day_of_year'] = dt.dt.dayofyear
    df_work['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df_work['month'] = dt.dt.month

    # 3. features
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

    # 4. data clean
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # 5. Extract features.
    feature_cols = [
        "open", "high", "low", "close", "volume", "open_interest",
        "day_of_week", "day_of_year", "week_of_year", "month",
        "ATR", "RSI", "roll_return"
    ]
    features = df_work[feature_cols].values

    # 6. Split the training and test sets.
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    # timestamps
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

    # Additional volatility statistics for interpretability discussion
    volatility_proxy = {
        "atr_mean": float(df_work["ATR"].iloc[train_size + gap:].mean()),
        "return_std": float(df_work["roll_return"].iloc[train_size + gap:].std())
    }

    return train_loader, test_loader, scaler, test_target_dates, len(feature_cols), volatility_proxy


# =============================== model ===============================
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation, groups=channels, bias=False
        )
        self.pointwise = nn.Conv1d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        y = self.depthwise(x)
        y = F.gelu(y)
        y = self.pointwise(y)
        y = self.bn(y)
        return y


class AuxiliaryStreams1(nn.Module):
    # moving average window
    def __init__(self, channels: int, seq_len: int, ma_kernel_size: int = 21):
        super().__init__()
        self.ma_kernel_size = ma_kernel_size
        self.conv = DepthwiseSeparableConv1d(channels, kernel_size=7, dilation=2)
        self.linear = nn.Linear(seq_len, PRED_LEN)

    def moving_avg(self, x):
        padding = (self.ma_kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=self.ma_kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        y = self.conv(seasonal.permute(0, 2, 1))
        y = y.squeeze(1)
        out = self.linear(y)
        return out


class AuxiliaryStreams2(nn.Module):
    def __init__(self, seq_len: int, dropout: float = 0.2, activation: str = 'GELU'):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)

        act_layer = nn.GELU() if activation == 'GELU' else nn.ReLU()

        self.net = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(seq_len // 2, PRED_LEN)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class DENet(nn.Module):
    """
    fusion_mode:
        - "learnable": beta and gamma are nn.Parameter
        - "fixed":     beta and gamma are fixed constants
    beta  -> weight of Auxiliary Stream I (auto_block)
    gamma -> weight of Auxiliary Stream II (nhits_block)
    """

    def __init__(
            self,
            seq_len: int,
            in_features: int,
            ma_kernel_size: int = 21,
            nhits_init: float = 0.01,  # gamma init
            auto_init: float = 0.2,  # beta init
            activation: str = 'GELU',
            fusion_mode: str = "learnable",
            fixed_beta: float = 1.0,
            fixed_gamma: float = 1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.ma_kernel_size = ma_kernel_size
        self.fusion_mode = fusion_mode

        self.trend_linear = nn.Linear(seq_len, PRED_LEN)
        self.residual_linear = nn.Linear(seq_len, PRED_LEN)
        self.feature_weights = nn.Parameter(torch.ones(in_features))

        self.decoder = nn.Sequential(
            nn.Linear(PRED_LEN, PRED_LEN * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(PRED_LEN * 2, PRED_LEN)
        )
        self.auto_block = AuxiliaryStreams1(
            channels=in_features,
            seq_len=seq_len,
            ma_kernel_size=ma_kernel_size
        )
        self.nhits_block = AuxiliaryStreams2(
            seq_len=seq_len,
            dropout=0.2,
            activation=activation
        )

        if fusion_mode == "learnable":
            self.beta = nn.Parameter(torch.tensor(auto_init, dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(nhits_init, dtype=torch.float32))
        elif fusion_mode == "fixed":
            self.register_buffer("beta", torch.tensor(float(fixed_beta), dtype=torch.float32))
            self.register_buffer("gamma", torch.tensor(float(fixed_gamma), dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    def moving_avg(self, x):
        padding = (self.ma_kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=self.ma_kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def get_fusion_weights(self):
        return float(self.beta.detach().cpu().item()), float(self.gamma.detach().cpu().item())

    def forward(self, x):  # x[B, S, in_features]
        w = self.feature_weights.unsqueeze(0).unsqueeze(0)
        weighted_x = x * w

        trend = self.moving_avg(weighted_x)
        residual = weighted_x - trend

        trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_linear(residual.permute(0, 2, 1)).permute(0, 2, 1)

        base_pred = (trend_pred + residual_pred)[:, :, 3]
        refined_base = base_pred + 0.1 * self.decoder(base_pred)

        auto_out = self.auto_block(weighted_x)  # Auxiliary Stream I
        nhits_out = self.nhits_block(residual[:, :, 3])  # Auxiliary Stream II

        out = refined_base + self.beta * auto_out + self.gamma * nhits_out
        return out


class DirectionalLoss(nn.Module):
    def __init__(self, alpha=2.0):  # Default alpha for Daily
        super().__init__()
        self.alpha = alpha
        self.base_loss = nn.SmoothL1Loss()

    def forward(self, pred, true, last_close):
        num_loss = self.base_loss(pred, true)

        # Ensure shapes align
        last_close = last_close.view(-1, 1)

        true_diff = true - last_close
        pred_diff = pred - last_close

        # Simple penalty for wrong direction
        penalty = F.relu(-1.0 * true_diff * pred_diff)

        return num_loss + self.alpha * torch.mean(penalty)


# ==============================================================
def denormalize_close(scaled_val, scaler, num_features):
    dummy = np.zeros((1, num_features))
    dummy[0, 3] = scaled_val  # Close
    denorm = scaler.inverse_transform(dummy)[0, 3]
    return np.exp(denorm) - 1e-8


def train_single_model(model, train_loader, val_loader, verbose_tag=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    criterion = DirectionalLoss(alpha=2.0)

    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training... {verbose_tag}")
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, last_close in val_loader:
                bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by, last_close).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            beta, gamma = model.get_fusion_weights()
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"beta: {beta:.4f} | gamma: {gamma:.4f}"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch + 1} {verbose_tag}")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    best_beta, best_gamma = model.get_fusion_weights()
    return model, best_loss, best_epoch, best_beta, best_gamma


def evaluate_model(model, test_loader, scaler, num_features):
    model.eval()

    preds_scaled, trues_scaled, prev_closes_scaled = [], [], []

    with torch.no_grad():
        for bx, by, b_prev_c in test_loader:
            bx = bx.to(device)
            output = model(bx)
            preds_scaled.extend(output.cpu().numpy().flatten())
            trues_scaled.extend(by.numpy().flatten())
            prev_closes_scaled.extend(b_prev_c.numpy().flatten())

    # Denormalization
    final_preds, final_trues, final_prev_closes = [], [], []
    loop_len = min(len(preds_scaled), len(trues_scaled), len(prev_closes_scaled))
    for i in range(loop_len):
        final_preds.append(denormalize_close(preds_scaled[i], scaler, num_features))
        final_trues.append(denormalize_close(trues_scaled[i], scaler, num_features))
        final_prev_closes.append(denormalize_close(prev_closes_scaled[i], scaler, num_features))

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

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MSE": float(mse),
        "R2": float(r2),
        "Trend_Acc": float(trend_acc),
        "preds": final_preds,
        "trues": final_trues,
        "prev_closes": final_prev_closes
    }
    return metrics


def build_model(num_features, fusion_mode="learnable", fixed_beta=1.0, fixed_gamma=1.0):
    model = DENet(
        seq_len=LOOK_BACK,
        in_features=num_features,
        ma_kernel_size=21,
        nhits_init=0.01,
        auto_init=0.2,
        activation='GELU',
        fusion_mode=fusion_mode,
        fixed_beta=fixed_beta,
        fixed_gamma=fixed_gamma
    ).to(device)
    return model


def save_predictions(test_target_dates, preds, output_file, suffix=""):
    base, ext = os.path.splitext(output_file)
    actual_output_file = f"{base}{suffix}{ext}"

    min_len = min(len(test_target_dates), len(preds))
    results_df = pd.DataFrame({
        "datetime": test_target_dates[:min_len],
        "pred_close": preds[:min_len]
    })
    os.makedirs(os.path.dirname(actual_output_file), exist_ok=True)
    results_df.to_csv(actual_output_file, index=False)
    print(f"Successfully saved predictions to: {actual_output_file}")
    return actual_output_file


def print_metrics(title, metrics, beta, gamma, best_epoch=None, best_val_loss=None):
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)
    if best_epoch is not None:
        print(f" Best Epoch   : {best_epoch}")
    if best_val_loss is not None:
        print(f" Best Val Loss: {best_val_loss:.6f}")
    print(f" beta         : {beta:.6f}")
    print(f" gamma        : {gamma:.6f}")
    print(f" MAE          : {metrics['MAE']:.4f}")
    print(f" RMSE         : {metrics['RMSE']:.4f}")
    print(f" MSE          : {metrics['MSE']:.4f}")
    print(f" R^2          : {metrics['R2']:.4f}")
    print(f" Trend Acc    : {metrics['Trend_Acc']:.2f}%")
    print("=" * 60 + "\n")


def run_learnable_experiment(asset_name, frequency, train_loader, test_loader, scaler, test_target_dates, num_features,
                             output_file):
    model = build_model(num_features, fusion_mode="learnable")
    model, best_val_loss, best_epoch, best_beta, best_gamma = train_single_model(
        model, train_loader, test_loader,
        verbose_tag=f"[{asset_name}-{frequency}] [learnable]"
    )
    metrics = evaluate_model(model, test_loader, scaler, num_features)
    save_predictions(test_target_dates, metrics["preds"], output_file, suffix="_learnable")
    print_metrics(
        title=f"{asset_name}-{frequency} | DENet (Learnable beta/gamma)",
        metrics=metrics,
        beta=best_beta,
        gamma=best_gamma,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss
    )

    result_row = {
        "asset_name": asset_name,
        "frequency": frequency,
        "model_variant": "DENet_learnable",
        "beta": best_beta,
        "gamma": best_gamma,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MSE": metrics["MSE"],
        "R2": metrics["R2"],
        "Trend_Acc": metrics["Trend_Acc"]
    }
    return result_row


def run_fixed_experiment(asset_name, frequency, train_loader, test_loader, scaler, test_target_dates, num_features,
                         output_file, fixed_beta, fixed_gamma, tag="fixed"):
    model = build_model(
        num_features,
        fusion_mode="fixed",
        fixed_beta=fixed_beta,
        fixed_gamma=fixed_gamma
    )
    model, best_val_loss, best_epoch, best_beta, best_gamma = train_single_model(
        model, train_loader, test_loader,
        verbose_tag=f"[{asset_name}-{frequency}] [{tag}]"
    )
    metrics = evaluate_model(model, test_loader, scaler, num_features)
    save_predictions(
        test_target_dates,
        metrics["preds"],
        output_file,
        suffix=f"_{tag}_b{fixed_beta}_g{fixed_gamma}"
    )
    print_metrics(
        title=f"{asset_name}-{frequency} | DENet (Fixed beta/gamma) [{tag}]",
        metrics=metrics,
        beta=best_beta,
        gamma=best_gamma,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss
    )

    result_row = {
        "asset_name": asset_name,
        "frequency": frequency,
        "model_variant": f"DENet_{tag}",
        "beta": best_beta,
        "gamma": best_gamma,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MSE": metrics["MSE"],
        "R2": metrics["R2"],
        "Trend_Acc": metrics["Trend_Acc"]
    }
    return result_row


def run_all_predefined_fixed_experiments(asset_name, frequency, train_loader, test_loader, scaler, test_target_dates,
                                         num_features, output_file):
    fixed_rows = []

    print(f"\nRunning predefined fixed-weight baselines for [{asset_name}-{frequency}] ...")
    print("These pairs are treated as independent fixed experiments.")
    print("No best-pair selection will be performed.\n")

    for beta, gamma in PREDEFINED_FIXED_PAIRS:
        tag = f"fixed_gridcase"
        row = run_fixed_experiment(
            asset_name=asset_name,
            frequency=frequency,
            train_loader=train_loader,
            test_loader=test_loader,
            scaler=scaler,
            test_target_dates=test_target_dates,
            num_features=num_features,
            output_file=output_file,
            fixed_beta=beta,
            fixed_gamma=gamma,
            tag=tag
        )
        row["grid_case_beta"] = beta
        row["grid_case_gamma"] = gamma
        fixed_rows.append(row)

    return fixed_rows


def run_process():
    all_results = []

    for exp_cfg in EXPERIMENTS:
        asset_name = exp_cfg["asset_name"]
        frequency = exp_cfg["frequency"]
        data_file = exp_cfg["data_file"]
        output_file = exp_cfg["output_file"]

        print(f"\nLoading data from {data_file} ...")
        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found. Skip.")
            continue

        df = pd.read_csv(data_file)
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 1. prepare data
        train_loader, test_loader, scaler, test_target_dates, num_features, volatility_proxy = preprocess_data(df,
                                                                                                               LOOK_BACK)
        print(f"Data prepared. Using {num_features} features. Test samples: {len(test_target_dates)}")
        print(
            f"[{asset_name}-{frequency}] Volatility proxy | "
            f"ATR mean: {volatility_proxy['atr_mean']:.6f}, "
            f"roll_return std: {volatility_proxy['return_std']:.6f}"
        )

        # 2. learnable experiment
        if ABLATION_MODE in ["learnable_only", "all"]:
            row = run_learnable_experiment(
                asset_name, frequency,
                train_loader, test_loader, scaler, test_target_dates, num_features, output_file
            )
            row["ATR_mean"] = volatility_proxy["atr_mean"]
            row["roll_return_std"] = volatility_proxy["return_std"]
            all_results.append(row)

        # 3. all predefined fixed experiments
        if ABLATION_MODE in ["fixed_only", "all"]:
            fixed_rows = run_all_predefined_fixed_experiments(
                asset_name, frequency,
                train_loader, test_loader, scaler, test_target_dates, num_features, output_file
            )
            for row in fixed_rows:
                row["ATR_mean"] = volatility_proxy["atr_mean"]
                row["roll_return_std"] = volatility_proxy["return_std"]
                all_results.append(row)

    # 4. save summary
    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)

        print("\n" + "=" * 100)
        print(" Ablation Summary ")
        print("=" * 100)

        show_cols = [
            "asset_name", "frequency", "model_variant",
            "beta", "gamma", "ATR_mean", "roll_return_std",
            "MAE", "RMSE", "R2", "Trend_Acc"
        ]
        existing_cols = [c for c in show_cols if c in summary_df.columns]
        print(summary_df[existing_cols].to_string(index=False))
        print("=" * 100)

        learnable_df = summary_df[summary_df["model_variant"] == "DENet_learnable"].copy()
        if len(learnable_df) > 0:
            print("\n" + "=" * 100)
            print(" Learned Fusion Weights (for interpretation) ")
            print("=" * 100)
            print(learnable_df[[
                "asset_name", "frequency", "beta", "gamma",
                "ATR_mean", "roll_return_std", "MAE", "Trend_Acc"
            ]].to_string(index=False))
            print("=" * 100)

        fixed_df = summary_df[summary_df["model_variant"].str.contains("fixed", na=False)].copy()
        if len(fixed_df) > 0:
            print("\n" + "=" * 100)
            print(" Fixed-Weight Cases Summary ")
            print("=" * 100)
            print(fixed_df[[
                "asset_name", "frequency", "beta", "gamma",
                "MAE", "RMSE", "R2", "Trend_Acc"
            ]].to_string(index=False))
            print("=" * 100)
    else:
        print("No experiment result generated.")


if __name__ == "__main__":
    run_process()
"""
D:\miniconda\envs\Py123\python.exe D:\桌面\FanGao\Final_Experiment\JM\daily_ablation\ab2.py 

Loading data from JM_daily.csv ...
Data prepared. Using 13 features. Test samples: 443
[JM-daily] Volatility proxy | ATR mean: 0.035554, roll_return std: 0.007171
Starting training... [JM-daily] [learnable]
Epoch 20/180 | Train Loss: 0.007255 | Val Loss: 0.005296 | beta: 0.1342 | gamma: 0.1036
Epoch 40/180 | Train Loss: 0.005236 | Val Loss: 0.004466 | beta: 0.1266 | gamma: 0.1056
Epoch 60/180 | Train Loss: 0.004353 | Val Loss: 0.003763 | beta: 0.1261 | gamma: 0.1119
Epoch 80/180 | Train Loss: 0.003919 | Val Loss: 0.003434 | beta: 0.1299 | gamma: 0.1157
Epoch 100/180 | Train Loss: 0.003876 | Val Loss: 0.003181 | beta: 0.1389 | gamma: 0.1184
Epoch 120/180 | Train Loss: 0.003482 | Val Loss: 0.003078 | beta: 0.1491 | gamma: 0.1223
Epoch 140/180 | Train Loss: 0.004281 | Val Loss: 0.003188 | beta: 0.1567 | gamma: 0.1242
Epoch 160/180 | Train Loss: 0.003399 | Val Loss: 0.003004 | beta: 0.1631 | gamma: 0.1243
Epoch 180/180 | Train Loss: 0.003215 | Val Loss: 0.002959 | beta: 0.1669 | gamma: 0.1246
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_learnable.csv

============================================================
 JM-daily | DENet (Learnable beta/gamma) 
============================================================
 Best Epoch   : 179
 Best Val Loss: 0.002952
 beta         : 0.166773
 gamma        : 0.124838
 MAE          : 28.6591
 RMSE         : 38.0026
 MSE          : 1444.1996
 R^2          : 0.9759
 Trend Acc    : 56.88%
============================================================


Running predefined fixed-weight baselines for [JM-daily] ...
These pairs are treated as independent fixed experiments.
No best-pair selection will be performed.

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.011047 | Val Loss: 0.007697 | beta: 1.0000 | gamma: 1.0000
Epoch 40/180 | Train Loss: 0.006361 | Val Loss: 0.005175 | beta: 1.0000 | gamma: 1.0000
Epoch 60/180 | Train Loss: 0.005004 | Val Loss: 0.004647 | beta: 1.0000 | gamma: 1.0000
Epoch 80/180 | Train Loss: 0.004511 | Val Loss: 0.003954 | beta: 1.0000 | gamma: 1.0000
Epoch 100/180 | Train Loss: 0.004265 | Val Loss: 0.003795 | beta: 1.0000 | gamma: 1.0000
Epoch 120/180 | Train Loss: 0.004053 | Val Loss: 0.003446 | beta: 1.0000 | gamma: 1.0000
Early stopping at epoch 135 [JM-daily] [fixed_gridcase]
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b1.0_g1.0.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 120
 Best Val Loss: 0.003446
 beta         : 1.000000
 gamma        : 1.000000
 MAE          : 29.7129
 RMSE         : 39.0198
 MSE          : 1522.5465
 R^2          : 0.9746
 Trend Acc    : 51.24%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.008504 | Val Loss: 0.008186 | beta: 0.0000 | gamma: 0.0000
Epoch 40/180 | Train Loss: 0.005870 | Val Loss: 0.006023 | beta: 0.0000 | gamma: 0.0000
Epoch 60/180 | Train Loss: 0.005084 | Val Loss: 0.004892 | beta: 0.0000 | gamma: 0.0000
Epoch 80/180 | Train Loss: 0.004434 | Val Loss: 0.004438 | beta: 0.0000 | gamma: 0.0000
Epoch 100/180 | Train Loss: 0.004142 | Val Loss: 0.004206 | beta: 0.0000 | gamma: 0.0000
Epoch 120/180 | Train Loss: 0.003736 | Val Loss: 0.003643 | beta: 0.0000 | gamma: 0.0000
Epoch 140/180 | Train Loss: 0.003495 | Val Loss: 0.003402 | beta: 0.0000 | gamma: 0.0000
Epoch 160/180 | Train Loss: 0.003318 | Val Loss: 0.003351 | beta: 0.0000 | gamma: 0.0000
Epoch 180/180 | Train Loss: 0.003494 | Val Loss: 0.003106 | beta: 0.0000 | gamma: 0.0000
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.0_g0.0.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 178
 Best Val Loss: 0.003095
 beta         : 0.000000
 gamma        : 0.000000
 MAE          : 29.0637
 RMSE         : 38.6550
 MSE          : 1494.2115
 R^2          : 0.9751
 Trend Acc    : 53.27%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.012458 | Val Loss: 0.012230 | beta: 0.0100 | gamma: 0.0100
Epoch 40/180 | Train Loss: 0.007279 | Val Loss: 0.006478 | beta: 0.0100 | gamma: 0.0100
Epoch 60/180 | Train Loss: 0.005608 | Val Loss: 0.005287 | beta: 0.0100 | gamma: 0.0100
Epoch 80/180 | Train Loss: 0.005785 | Val Loss: 0.004459 | beta: 0.0100 | gamma: 0.0100
Epoch 100/180 | Train Loss: 0.004450 | Val Loss: 0.004026 | beta: 0.0100 | gamma: 0.0100
Epoch 120/180 | Train Loss: 0.003986 | Val Loss: 0.003743 | beta: 0.0100 | gamma: 0.0100
Epoch 140/180 | Train Loss: 0.003840 | Val Loss: 0.003592 | beta: 0.0100 | gamma: 0.0100
Epoch 160/180 | Train Loss: 0.003604 | Val Loss: 0.003488 | beta: 0.0100 | gamma: 0.0100
Epoch 180/180 | Train Loss: 0.003359 | Val Loss: 0.003243 | beta: 0.0100 | gamma: 0.0100
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.01_g0.01.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 180
 Best Val Loss: 0.003243
 beta         : 0.010000
 gamma        : 0.010000
 MAE          : 29.5588
 RMSE         : 38.7845
 MSE          : 1504.2355
 R^2          : 0.9749
 Trend Acc    : 51.92%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.007156 | Val Loss: 0.006510 | beta: 0.0500 | gamma: 0.0500
Epoch 40/180 | Train Loss: 0.005031 | Val Loss: 0.004732 | beta: 0.0500 | gamma: 0.0500
Epoch 60/180 | Train Loss: 0.004448 | Val Loss: 0.004154 | beta: 0.0500 | gamma: 0.0500
Epoch 80/180 | Train Loss: 0.003715 | Val Loss: 0.003573 | beta: 0.0500 | gamma: 0.0500
Epoch 100/180 | Train Loss: 0.003477 | Val Loss: 0.003311 | beta: 0.0500 | gamma: 0.0500
Epoch 120/180 | Train Loss: 0.003570 | Val Loss: 0.003233 | beta: 0.0500 | gamma: 0.0500
Epoch 140/180 | Train Loss: 0.003625 | Val Loss: 0.003203 | beta: 0.0500 | gamma: 0.0500
Epoch 160/180 | Train Loss: 0.003134 | Val Loss: 0.003004 | beta: 0.0500 | gamma: 0.0500
Epoch 180/180 | Train Loss: 0.002953 | Val Loss: 0.003179 | beta: 0.0500 | gamma: 0.0500
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.05_g0.05.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 174
 Best Val Loss: 0.002958
 beta         : 0.050000
 gamma        : 0.050000
 MAE          : 29.1247
 RMSE         : 38.4070
 MSE          : 1475.0970
 R^2          : 0.9754
 Trend Acc    : 53.50%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.009515 | Val Loss: 0.008609 | beta: 0.1000 | gamma: 0.0100
Epoch 40/180 | Train Loss: 0.006634 | Val Loss: 0.005762 | beta: 0.1000 | gamma: 0.0100
Epoch 60/180 | Train Loss: 0.005612 | Val Loss: 0.004827 | beta: 0.1000 | gamma: 0.0100
Epoch 80/180 | Train Loss: 0.004516 | Val Loss: 0.004118 | beta: 0.1000 | gamma: 0.0100
Epoch 100/180 | Train Loss: 0.004081 | Val Loss: 0.003626 | beta: 0.1000 | gamma: 0.0100
Epoch 120/180 | Train Loss: 0.003756 | Val Loss: 0.003395 | beta: 0.1000 | gamma: 0.0100
Epoch 140/180 | Train Loss: 0.003373 | Val Loss: 0.003216 | beta: 0.1000 | gamma: 0.0100
Epoch 160/180 | Train Loss: 0.003538 | Val Loss: 0.002991 | beta: 0.1000 | gamma: 0.0100
Epoch 180/180 | Train Loss: 0.003154 | Val Loss: 0.002991 | beta: 0.1000 | gamma: 0.0100
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.1_g0.01.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 175
 Best Val Loss: 0.002890
 beta         : 0.100000
 gamma        : 0.010000
 MAE          : 28.9254
 RMSE         : 38.1477
 MSE          : 1455.2433
 R^2          : 0.9757
 Trend Acc    : 54.18%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.013122 | Val Loss: 0.010065 | beta: 0.1000 | gamma: 0.0500
Epoch 40/180 | Train Loss: 0.007547 | Val Loss: 0.006132 | beta: 0.1000 | gamma: 0.0500
Epoch 60/180 | Train Loss: 0.005957 | Val Loss: 0.004687 | beta: 0.1000 | gamma: 0.0500
Epoch 80/180 | Train Loss: 0.004646 | Val Loss: 0.004007 | beta: 0.1000 | gamma: 0.0500
Epoch 100/180 | Train Loss: 0.003931 | Val Loss: 0.003610 | beta: 0.1000 | gamma: 0.0500
Epoch 120/180 | Train Loss: 0.003741 | Val Loss: 0.003564 | beta: 0.1000 | gamma: 0.0500
Epoch 140/180 | Train Loss: 0.003972 | Val Loss: 0.003437 | beta: 0.1000 | gamma: 0.0500
Epoch 160/180 | Train Loss: 0.003239 | Val Loss: 0.003156 | beta: 0.1000 | gamma: 0.0500
Epoch 180/180 | Train Loss: 0.003241 | Val Loss: 0.003100 | beta: 0.1000 | gamma: 0.0500
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.1_g0.05.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 179
 Best Val Loss: 0.003062
 beta         : 0.100000
 gamma        : 0.050000
 MAE          : 29.3676
 RMSE         : 38.5031
 MSE          : 1482.4887
 R^2          : 0.9753
 Trend Acc    : 49.21%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.007750 | Val Loss: 0.006748 | beta: 0.1000 | gamma: 0.1000
Epoch 40/180 | Train Loss: 0.005556 | Val Loss: 0.004779 | beta: 0.1000 | gamma: 0.1000
Epoch 60/180 | Train Loss: 0.004400 | Val Loss: 0.003969 | beta: 0.1000 | gamma: 0.1000
Epoch 80/180 | Train Loss: 0.004054 | Val Loss: 0.003801 | beta: 0.1000 | gamma: 0.1000
Epoch 100/180 | Train Loss: 0.003512 | Val Loss: 0.003300 | beta: 0.1000 | gamma: 0.1000
Epoch 120/180 | Train Loss: 0.003560 | Val Loss: 0.003186 | beta: 0.1000 | gamma: 0.1000
Epoch 140/180 | Train Loss: 0.003336 | Val Loss: 0.003177 | beta: 0.1000 | gamma: 0.1000
Epoch 160/180 | Train Loss: 0.003047 | Val Loss: 0.003018 | beta: 0.1000 | gamma: 0.1000
Epoch 180/180 | Train Loss: 0.003029 | Val Loss: 0.002985 | beta: 0.1000 | gamma: 0.1000
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.1_g0.1.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 176
 Best Val Loss: 0.002927
 beta         : 0.100000
 gamma        : 0.100000
 MAE          : 29.0713
 RMSE         : 38.5807
 MSE          : 1488.4713
 R^2          : 0.9752
 Trend Acc    : 51.47%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.012515 | Val Loss: 0.009252 | beta: 0.2000 | gamma: 0.0100
Epoch 40/180 | Train Loss: 0.008964 | Val Loss: 0.006662 | beta: 0.2000 | gamma: 0.0100
Epoch 60/180 | Train Loss: 0.005563 | Val Loss: 0.005939 | beta: 0.2000 | gamma: 0.0100
Epoch 80/180 | Train Loss: 0.004647 | Val Loss: 0.004731 | beta: 0.2000 | gamma: 0.0100
Epoch 100/180 | Train Loss: 0.004604 | Val Loss: 0.004219 | beta: 0.2000 | gamma: 0.0100
Epoch 120/180 | Train Loss: 0.003955 | Val Loss: 0.003927 | beta: 0.2000 | gamma: 0.0100
Epoch 140/180 | Train Loss: 0.003883 | Val Loss: 0.003762 | beta: 0.2000 | gamma: 0.0100
Epoch 160/180 | Train Loss: 0.004101 | Val Loss: 0.003778 | beta: 0.2000 | gamma: 0.0100
Epoch 180/180 | Train Loss: 0.003474 | Val Loss: 0.003328 | beta: 0.2000 | gamma: 0.0100
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.2_g0.01.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 178
 Best Val Loss: 0.003321
 beta         : 0.200000
 gamma        : 0.010000
 MAE          : 29.3896
 RMSE         : 38.9574
 MSE          : 1517.6805
 R^2          : 0.9747
 Trend Acc    : 53.27%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.008307 | Val Loss: 0.007332 | beta: 0.2000 | gamma: 0.0500
Epoch 40/180 | Train Loss: 0.007206 | Val Loss: 0.005629 | beta: 0.2000 | gamma: 0.0500
Epoch 60/180 | Train Loss: 0.005521 | Val Loss: 0.004974 | beta: 0.2000 | gamma: 0.0500
Epoch 80/180 | Train Loss: 0.004239 | Val Loss: 0.004334 | beta: 0.2000 | gamma: 0.0500
Epoch 100/180 | Train Loss: 0.004098 | Val Loss: 0.003967 | beta: 0.2000 | gamma: 0.0500
Epoch 120/180 | Train Loss: 0.003765 | Val Loss: 0.003620 | beta: 0.2000 | gamma: 0.0500
Epoch 140/180 | Train Loss: 0.003877 | Val Loss: 0.003492 | beta: 0.2000 | gamma: 0.0500
Epoch 160/180 | Train Loss: 0.003292 | Val Loss: 0.003356 | beta: 0.2000 | gamma: 0.0500
Epoch 180/180 | Train Loss: 0.003447 | Val Loss: 0.003377 | beta: 0.2000 | gamma: 0.0500
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.2_g0.05.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 175
 Best Val Loss: 0.003254
 beta         : 0.200000
 gamma        : 0.050000
 MAE          : 29.7651
 RMSE         : 39.0679
 MSE          : 1526.3023
 R^2          : 0.9746
 Trend Acc    : 47.40%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.009869 | Val Loss: 0.007291 | beta: 0.2000 | gamma: 0.1000
Epoch 40/180 | Train Loss: 0.006215 | Val Loss: 0.005348 | beta: 0.2000 | gamma: 0.1000
Epoch 60/180 | Train Loss: 0.005480 | Val Loss: 0.004351 | beta: 0.2000 | gamma: 0.1000
Epoch 80/180 | Train Loss: 0.005432 | Val Loss: 0.004074 | beta: 0.2000 | gamma: 0.1000
Epoch 100/180 | Train Loss: 0.004596 | Val Loss: 0.003921 | beta: 0.2000 | gamma: 0.1000
Epoch 120/180 | Train Loss: 0.004380 | Val Loss: 0.003820 | beta: 0.2000 | gamma: 0.1000
Epoch 140/180 | Train Loss: 0.004228 | Val Loss: 0.003648 | beta: 0.2000 | gamma: 0.1000
Epoch 160/180 | Train Loss: 0.004272 | Val Loss: 0.003574 | beta: 0.2000 | gamma: 0.1000
Epoch 180/180 | Train Loss: 0.004609 | Val Loss: 0.003673 | beta: 0.2000 | gamma: 0.1000
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.2_g0.1.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 175
 Best Val Loss: 0.003536
 beta         : 0.200000
 gamma        : 0.100000
 MAE          : 30.0679
 RMSE         : 39.5030
 MSE          : 1560.4874
 R^2          : 0.9740
 Trend Acc    : 50.11%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.007428 | Val Loss: 0.006304 | beta: 0.2000 | gamma: 0.2000
Epoch 40/180 | Train Loss: 0.005197 | Val Loss: 0.004395 | beta: 0.2000 | gamma: 0.2000
Epoch 60/180 | Train Loss: 0.004839 | Val Loss: 0.003898 | beta: 0.2000 | gamma: 0.2000
Epoch 80/180 | Train Loss: 0.004146 | Val Loss: 0.003695 | beta: 0.2000 | gamma: 0.2000
Epoch 100/180 | Train Loss: 0.004682 | Val Loss: 0.003927 | beta: 0.2000 | gamma: 0.2000
Epoch 120/180 | Train Loss: 0.003313 | Val Loss: 0.003353 | beta: 0.2000 | gamma: 0.2000
Epoch 140/180 | Train Loss: 0.003471 | Val Loss: 0.003253 | beta: 0.2000 | gamma: 0.2000
Epoch 160/180 | Train Loss: 0.003212 | Val Loss: 0.003226 | beta: 0.2000 | gamma: 0.2000
Early stopping at epoch 174 [JM-daily] [fixed_gridcase]
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.2_g0.2.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 159
 Best Val Loss: 0.003187
 beta         : 0.200000
 gamma        : 0.200000
 MAE          : 29.5121
 RMSE         : 39.0621
 MSE          : 1525.8498
 R^2          : 0.9746
 Trend Acc    : 48.53%
============================================================

Starting training... [JM-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.012073 | Val Loss: 0.008360 | beta: 0.5000 | gamma: 0.1000
Epoch 40/180 | Train Loss: 0.006328 | Val Loss: 0.005340 | beta: 0.5000 | gamma: 0.1000
Epoch 60/180 | Train Loss: 0.004973 | Val Loss: 0.004337 | beta: 0.5000 | gamma: 0.1000
Epoch 80/180 | Train Loss: 0.004805 | Val Loss: 0.004138 | beta: 0.5000 | gamma: 0.1000
Epoch 100/180 | Train Loss: 0.004037 | Val Loss: 0.003608 | beta: 0.5000 | gamma: 0.1000
Epoch 120/180 | Train Loss: 0.003795 | Val Loss: 0.003678 | beta: 0.5000 | gamma: 0.1000
Epoch 140/180 | Train Loss: 0.003798 | Val Loss: 0.003355 | beta: 0.5000 | gamma: 0.1000
Epoch 160/180 | Train Loss: 0.003402 | Val Loss: 0.003332 | beta: 0.5000 | gamma: 0.1000
Epoch 180/180 | Train Loss: 0.003830 | Val Loss: 0.003428 | beta: 0.5000 | gamma: 0.1000
Successfully saved predictions to: ../daily_ablation/pred_daily_advanced_fixed_gridcase_b0.5_g0.1.csv

============================================================
 JM-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 179
 Best Val Loss: 0.003283
 beta         : 0.500000
 gamma        : 0.100000
 MAE          : 29.4642
 RMSE         : 39.1195
 MSE          : 1530.3378
 R^2          : 0.9745
 Trend Acc    : 50.34%
============================================================


====================================================================================================
 Ablation Summary 
====================================================================================================
asset_name frequency        model_variant     beta    gamma  ATR_mean  roll_return_std       MAE      RMSE       R2  Trend_Acc
        JM     daily      DENet_learnable 0.166773 0.124838  0.035554         0.007171 28.659088 38.002626 0.975932  56.884876
        JM     daily DENet_fixed_gridcase 1.000000 1.000000  0.035554         0.007171 29.712893 39.019822 0.974626  51.241535
        JM     daily DENet_fixed_gridcase 0.000000 0.000000  0.035554         0.007171 29.063656 38.655032 0.975098  53.273138
        JM     daily DENet_fixed_gridcase 0.010000 0.010000  0.035554         0.007171 29.558830 38.784475 0.974931  51.918736
        JM     daily DENet_fixed_gridcase 0.050000 0.050000  0.035554         0.007171 29.124718 38.406991 0.975417  53.498871
        JM     daily DENet_fixed_gridcase 0.100000 0.010000  0.035554         0.007171 28.925354 38.147652 0.975748  54.176072
        JM     daily DENet_fixed_gridcase 0.100000 0.050000  0.035554         0.007171 29.367619 38.503099 0.975294  49.209932
        JM     daily DENet_fixed_gridcase 0.100000 0.100000  0.035554         0.007171 29.071250 38.580712 0.975194  51.467269
        JM     daily DENet_fixed_gridcase 0.200000 0.010000  0.035554         0.007171 29.389574 38.957419 0.974707  53.273138
        JM     daily DENet_fixed_gridcase 0.200000 0.050000  0.035554         0.007171 29.765093 39.067920 0.974563  47.404063
        JM     daily DENet_fixed_gridcase 0.200000 0.100000  0.035554         0.007171 30.067926 39.503006 0.973994  50.112867
        JM     daily DENet_fixed_gridcase 0.200000 0.200000  0.035554         0.007171 29.512106 39.062127 0.974571  48.532731
        JM     daily DENet_fixed_gridcase 0.500000 0.100000  0.035554         0.007171 29.464181 39.119532 0.974496  50.338600
====================================================================================================

====================================================================================================
 Learned Fusion Weights (for interpretation) 
====================================================================================================
asset_name frequency     beta    gamma  ATR_mean  roll_return_std       MAE  Trend_Acc
        JM     daily 0.166773 0.124838  0.035554         0.007171 28.659088  56.884876
====================================================================================================

====================================================================================================
 Fixed-Weight Cases Summary 
====================================================================================================
asset_name frequency  beta  gamma       MAE      RMSE       R2  Trend_Acc
        JM     daily  1.00   1.00 29.712893 39.019822 0.974626  51.241535
        JM     daily  0.00   0.00 29.063656 38.655032 0.975098  53.273138
        JM     daily  0.01   0.01 29.558830 38.784475 0.974931  51.918736
        JM     daily  0.05   0.05 29.124718 38.406991 0.975417  53.498871
        JM     daily  0.10   0.01 28.925354 38.147652 0.975748  54.176072
        JM     daily  0.10   0.05 29.367619 38.503099 0.975294  49.209932
        JM     daily  0.10   0.10 29.071250 38.580712 0.975194  51.467269
        JM     daily  0.20   0.01 29.389574 38.957419 0.974707  53.273138
        JM     daily  0.20   0.05 29.765093 39.067920 0.974563  47.404063
        JM     daily  0.20   0.10 30.067926 39.503006 0.973994  50.112867
        JM     daily  0.20   0.20 29.512106 39.062127 0.974571  48.532731
        JM     daily  0.50   0.10 29.464181 39.119532 0.974496  50.338600
====================================================================================================

进程已结束，退出代码为 0

"""