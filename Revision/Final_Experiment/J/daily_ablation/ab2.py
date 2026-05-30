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
LR = 5e-3
BATCH_SIZE = 64
DATA_FILE = "J_daily.csv"
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
    (1.00,1.00),
    (0.01,0.01),
    (0.05,0.05),
    (0.10,0.01),
    (0.10,0.05),
    (0.10,0.10),
    (0.20,0.01),
    (0.20,0.05),
    (0.20,0.20),
    (0.50,0.10),
    (0.00,1.00),
    (0.01,0.05),
    (0.10,0.20),
    (0.05,0.50),
    (0.10,0.20),
    (0.20, 0.10),

]
""""""
# ----------------------------------------------------------------
# Multi-asset / multi-frequency experiment list
EXPERIMENTS = [
    {
        "asset_name": "J",
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
        nhits_init: float = 0.01,   # gamma init
        auto_init: float = 0.2,     # beta init
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

        auto_out = self.auto_block(weighted_x)          # Auxiliary Stream I
        nhits_out = self.nhits_block(residual[:, :, 3]) # Auxiliary Stream II

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


def run_learnable_experiment(asset_name, frequency, train_loader, test_loader, scaler, test_target_dates, num_features, output_file):
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


def run_fixed_experiment(asset_name, frequency, train_loader, test_loader, scaler, test_target_dates, num_features, output_file, fixed_beta, fixed_gamma, tag="fixed"):
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


def run_all_predefined_fixed_experiments(asset_name, frequency, train_loader, test_loader, scaler, test_target_dates, num_features, output_file):
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
        train_loader, test_loader, scaler, test_target_dates, num_features, volatility_proxy = preprocess_data(df, LOOK_BACK)
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

