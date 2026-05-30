import os
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
DATA_FILE = "I_daily.csv"


# ----------------------------------------------------------------
# Ablation settings
# "learnable_only" -> only current DENet
# "fixed_only"     -> only run all predefined fixed (beta, gamma) combinations
# "init_grid_only" -> only run learnable-init grid
# "all"            -> run learnable + fixed grid + learnable-init grid
ABLATION_MODE = "all"

# ----------------------------------------------------------------
# Predefined fixed fusion weights
PREDEFINED_FIXED_PAIRS = [
    (1.0, 1.0),
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
# NEW: learnable beta/gamma initialization grid
LEARNABLE_INIT_GRID = {
    "beta": [0.01, 0.05, 0.08, 0.1],
    "gamma": [0.01, 0.02, 0.05,0.1,0.2]
}

# ----------------------------------------------------------------
EXPERIMENTS = [
    {
        "asset_name": "I",
        "frequency": "daily",
        "data_file": DATA_FILE,

    },
]

ABLATION_RESULT_FILE = "../daily_ablation/ablation_fusion_weights_summary.csv"


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

    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    dt = df_work['datetime']
    df_work['day_of_week'] = dt.dt.dayofweek
    df_work['day_of_year'] = dt.dt.dayofyear
    df_work['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df_work['month'] = dt.dt.month

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

    feature_cols = [
        "open", "high", "low", "close", "volume", "open_interest",
        "day_of_week", "day_of_year", "week_of_year", "month",
        "ATR", "RSI", "roll_return"
    ]
    features = df_work[feature_cols].values

    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    test_start_idx = train_size + gap
    num_test_samples = len(raw_test) - look_back - PRED_LEN + 1
    target_indices = [test_start_idx + i + look_back for i in range(num_test_samples)]
    test_target_dates = df_work['datetime'].iloc[target_indices].values

    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    train_ds = TimeSeriesDataset(train_data, seq_len=look_back)
    test_ds = TimeSeriesDataset(test_data, seq_len=look_back)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    volatility_proxy = {
        "atr_mean": float(df_work["ATR"].iloc[train_size + gap:].mean()),
        "return_std": float(df_work["roll_return"].iloc[train_size + gap:].std())
    }

    return train_loader, test_loader, scaler, test_target_dates, len(feature_cols), volatility_proxy


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
        nhits_init: float = 0.01,
        auto_init: float = 0.2,
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

    def forward(self, x):
        w = self.feature_weights.unsqueeze(0).unsqueeze(0)
        weighted_x = x * w

        trend = self.moving_avg(weighted_x)
        residual = weighted_x - trend

        trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_linear(residual.permute(0, 2, 1)).permute(0, 2, 1)

        base_pred = (trend_pred + residual_pred)[:, :, 3]
        refined_base = base_pred + 0.1 * self.decoder(base_pred)

        auto_out = self.auto_block(weighted_x)
        nhits_out = self.nhits_block(residual[:, :, 3])

        out = refined_base + self.beta * auto_out + self.gamma * nhits_out
        return out


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

        penalty = F.relu(-1.0 * true_diff * pred_diff)
        return num_loss + self.alpha * torch.mean(penalty)


def denormalize_close(scaled_val, scaler, num_features):
    dummy = np.zeros((1, num_features))
    dummy[0, 3] = scaled_val
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


def build_model(
    num_features,
    fusion_mode="learnable",
    fixed_beta=1.0,
    fixed_gamma=1.0,
    beta_init=0.2,
    gamma_init=0.01
):
    model = DENet(
        seq_len=LOOK_BACK,
        in_features=num_features,
        ma_kernel_size=21,
        nhits_init=gamma_init,
        auto_init=beta_init,
        activation='GELU',
        fusion_mode=fusion_mode,
        fixed_beta=fixed_beta,
        fixed_gamma=fixed_gamma
    ).to(device)
    return model


def save_predictions(test_target_dates, preds,  suffix=""):


    min_len = min(len(test_target_dates), len(preds))
    results_df = pd.DataFrame({
        "datetime": test_target_dates[:min_len],
        "pred_close": preds[:min_len]
    })
    


def print_metrics(title, metrics, beta, gamma, best_epoch=None, best_val_loss=None,
                  init_beta=None, init_gamma=None):
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)
    if init_beta is not None and init_gamma is not None:
        print(f" Init beta    : {init_beta:.6f}")
        print(f" Init gamma   : {init_gamma:.6f}")
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


def run_learnable_experiment(asset_name, frequency, train_loader, test_loader, scaler,
                             test_target_dates, num_features):
    model = build_model(num_features, fusion_mode="learnable", beta_init=0.2, gamma_init=0.01)
    model, best_val_loss, best_epoch, best_beta, best_gamma = train_single_model(
        model, train_loader, test_loader,
        verbose_tag=f"[{asset_name}-{frequency}] [learnable]"
    )
    metrics = evaluate_model(model, test_loader, scaler, num_features)
    save_predictions(test_target_dates, metrics["preds"], suffix="_learnable")

    print_metrics(
        title=f"{asset_name}-{frequency} | DENet (Learnable beta/gamma)",
        metrics=metrics,
        beta=best_beta,
        gamma=best_gamma,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        init_beta=0.2,
        init_gamma=0.01
    )

    result_row = {
        "asset_name": asset_name,
        "frequency": frequency,
        "model_variant": "DENet_learnable",
        "init_beta": 0.2,
        "init_gamma": 0.01,
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


def run_fixed_experiment(asset_name, frequency, train_loader, test_loader, scaler,
                         test_target_dates, num_features, 
                         fixed_beta, fixed_gamma, tag="fixed"):
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
        "init_beta": np.nan,
        "init_gamma": np.nan,
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


def run_all_predefined_fixed_experiments(asset_name, frequency, train_loader, test_loader, scaler,
                                         test_target_dates, num_features, ):
    fixed_rows = []

    print(f"\nRunning predefined fixed-weight baselines for [{asset_name}-{frequency}] ...")
    print("These pairs are treated as independent fixed experiments.")
    print("No best-pair selection will be performed.\n")

    for beta, gamma in PREDEFINED_FIXED_PAIRS:
        tag = "fixed_gridcase"
        row = run_fixed_experiment(
            asset_name=asset_name,
            frequency=frequency,
            train_loader=train_loader,
            test_loader=test_loader,
            scaler=scaler,
            test_target_dates=test_target_dates,
            num_features=num_features,
            fixed_beta=beta,
            fixed_gamma=gamma,
            tag=tag
        )
        row["grid_case_beta"] = beta
        row["grid_case_gamma"] = gamma
        fixed_rows.append(row)

    return fixed_rows


# ============================================================
# NEW: learnable init grid experiment
# ============================================================
def run_learnable_init_grid_experiment(asset_name, frequency, train_loader, test_loader, scaler,
                                       test_target_dates, num_features, 
                                       init_beta, init_gamma):
    model = build_model(
        num_features,
        fusion_mode="learnable",
        beta_init=init_beta,
        gamma_init=init_gamma
    )

    model, best_val_loss, best_epoch, best_beta, best_gamma = train_single_model(
        model, train_loader, test_loader,
        verbose_tag=f"[{asset_name}-{frequency}] [learnable_init_grid b0={init_beta}, g0={init_gamma}]"
    )

    metrics = evaluate_model(model, test_loader, scaler, num_features)

    save_predictions(
        test_target_dates,
        metrics["preds"],
        
        suffix=f"_learnable_initgrid_b0_{init_beta}_g0_{init_gamma}"
    )

    print_metrics(
        title=f"{asset_name}-{frequency} | DENet (Learnable init-grid)",
        metrics=metrics,
        beta=best_beta,
        gamma=best_gamma,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        init_beta=init_beta,
        init_gamma=init_gamma
    )

    result_row = {
        "asset_name": asset_name,
        "frequency": frequency,
        "model_variant": "DENet_learnable_initgrid",
        "init_beta": init_beta,
        "init_gamma": init_gamma,
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


def run_all_learnable_init_grid_experiments(asset_name, frequency, train_loader, test_loader, scaler,
                                            test_target_dates, num_features,):
    init_rows = []

    beta_inits = LEARNABLE_INIT_GRID["beta"]
    gamma_inits = LEARNABLE_INIT_GRID["gamma"]

    print(f"\nRunning learnable-init grid for [{asset_name}-{frequency}] ...")
    print("beta/gamma remain learnable during training; only initialization is changed.\n")

    for init_beta in beta_inits:
        for init_gamma in gamma_inits:
            row = run_learnable_init_grid_experiment(
                asset_name=asset_name,
                frequency=frequency,
                train_loader=train_loader,
                test_loader=test_loader,
                scaler=scaler,
                test_target_dates=test_target_dates,
                num_features=num_features,
                init_beta=init_beta,
                init_gamma=init_gamma
            )
            init_rows.append(row)

    return init_rows


def run_process():
    all_results = []

    for exp_cfg in EXPERIMENTS:
        asset_name = exp_cfg["asset_name"]
        frequency = exp_cfg["frequency"]
        data_file = exp_cfg["data_file"]


        print(f"\nLoading data from {data_file} ...")
        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found. Skip.")
            continue

        df = pd.read_csv(data_file)
        df["datetime"] = pd.to_datetime(df["datetime"])

        train_loader, test_loader, scaler, test_target_dates, num_features, volatility_proxy = preprocess_data(df, LOOK_BACK)
        print(f"Data prepared. Using {num_features} features. Test samples: {len(test_target_dates)}")
        print(
            f"[{asset_name}-{frequency}] Volatility proxy | "
            f"ATR mean: {volatility_proxy['atr_mean']:.6f}, "
            f"roll_return std: {volatility_proxy['return_std']:.6f}"
        )

        if ABLATION_MODE in ["learnable_only", "all"]:
            row = run_learnable_experiment(
                asset_name, frequency,
                train_loader, test_loader, scaler, test_target_dates, num_features,
            )
            row["ATR_mean"] = volatility_proxy["atr_mean"]
            row["roll_return_std"] = volatility_proxy["return_std"]
            all_results.append(row)

        if ABLATION_MODE in ["fixed_only", "all"]:
            fixed_rows = run_all_predefined_fixed_experiments(
                asset_name, frequency,
                train_loader, test_loader, scaler, test_target_dates, num_features,
            )
            for row in fixed_rows:
                row["ATR_mean"] = volatility_proxy["atr_mean"]
                row["roll_return_std"] = volatility_proxy["return_std"]
                all_results.append(row)

        if ABLATION_MODE in ["init_grid_only", "all"]:
            init_rows = run_all_learnable_init_grid_experiments(
                asset_name, frequency,
                train_loader, test_loader, scaler, test_target_dates, num_features,
            )
            for row in init_rows:
                row["ATR_mean"] = volatility_proxy["atr_mean"]
                row["roll_return_std"] = volatility_proxy["return_std"]
                all_results.append(row)

    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)
        os.makedirs(os.path.dirname(ABLATION_RESULT_FILE), exist_ok=True)
        summary_df.to_csv(ABLATION_RESULT_FILE, index=False)
        print(f"\nAblation summary saved to: {ABLATION_RESULT_FILE}")

        print("\n" + "=" * 120)
        print(" Ablation Summary ")
        print("=" * 120)

        show_cols = [
            "asset_name", "frequency", "model_variant",
            "init_beta", "init_gamma",
            "beta", "gamma",
            "ATR_mean", "roll_return_std",
            "MAE", "RMSE", "R2", "Trend_Acc"
        ]
        existing_cols = [c for c in show_cols if c in summary_df.columns]
        print(summary_df[existing_cols].to_string(index=False))
        print("=" * 120)

        learnable_df = summary_df[summary_df["model_variant"] == "DENet_learnable"].copy()
        if len(learnable_df) > 0:
            print("\n" + "=" * 120)
            print(" Learned Fusion Weights (Default init) ")
            print("=" * 120)
            print(learnable_df[[
                "asset_name", "frequency",
                "init_beta", "init_gamma",
                "beta", "gamma",
                "ATR_mean", "roll_return_std",
                "MAE", "Trend_Acc"
            ]].to_string(index=False))
            print("=" * 120)

        initgrid_df = summary_df[summary_df["model_variant"] == "DENet_learnable_initgrid"].copy()
        if len(initgrid_df) > 0:
            print("\n" + "=" * 120)
            print(" Learnable Init Grid Summary ")
            print("=" * 120)
            print(initgrid_df[[
                "asset_name", "frequency",
                "init_beta", "init_gamma",
                "beta", "gamma",
                "MAE", "RMSE", "R2", "Trend_Acc"
            ]].sort_values(by=["MAE", "RMSE"]).to_string(index=False))
            print("=" * 120)

        fixed_df = summary_df[summary_df["model_variant"].str.contains("fixed", na=False)].copy()
        if len(fixed_df) > 0:
            print("\n" + "=" * 120)
            print(" Fixed-Weight Cases Summary ")
            print("=" * 120)
            print(fixed_df[[
                "asset_name", "frequency", "beta", "gamma",
                "MAE", "RMSE", "R2", "Trend_Acc"
            ]].to_string(index=False))
            print("=" * 120)
    else:
        print("No experiment result generated.")


if __name__ == "__main__":
    run_process()

"""
D:\miniconda\envs\Py123\python.exe D:\桌面\FanGao\Final_Experiment\SM\daily_ablation\ab2.py 

Loading data from I_daily.csv ...
Data prepared. Using 13 features. Test samples: 443
[I-daily] Volatility proxy | ATR mean: 0.028750, roll_return std: 0.006949
Starting training... [I-daily] [learnable]
Epoch 20/180 | Train Loss: 0.004642 | Val Loss: 0.002349 | beta: 0.1256 | gamma: 0.0989
Epoch 40/180 | Train Loss: 0.003196 | Val Loss: 0.001694 | beta: 0.1109 | gamma: 0.1037
Epoch 60/180 | Train Loss: 0.002412 | Val Loss: 0.001426 | beta: 0.1016 | gamma: 0.1099
Epoch 80/180 | Train Loss: 0.002327 | Val Loss: 0.001310 | beta: 0.0927 | gamma: 0.1190
Epoch 100/180 | Train Loss: 0.002156 | Val Loss: 0.001409 | beta: 0.0905 | gamma: 0.1230
Epoch 120/180 | Train Loss: 0.001987 | Val Loss: 0.001302 | beta: 0.0953 | gamma: 0.1269
Epoch 140/180 | Train Loss: 0.001971 | Val Loss: 0.001157 | beta: 0.1025 | gamma: 0.1279
Epoch 160/180 | Train Loss: 0.002028 | Val Loss: 0.001155 | beta: 0.1127 | gamma: 0.1290
Epoch 180/180 | Train Loss: 0.001857 | Val Loss: 0.001121 | beta: 0.1194 | gamma: 0.1281

============================================================
 I-daily | DENet (Learnable beta/gamma) 
============================================================
 Init beta    : 0.200000
 Init gamma   : 0.010000
 Best Epoch   : 172
 Best Val Loss: 0.001100
 beta         : 0.116179
 gamma        : 0.128128
 MAE          : 12.4277
 RMSE         : 16.6054
 MSE          : 275.7387
 R^2          : 0.9575
 Trend Acc    : 50.11%
============================================================


Running predefined fixed-weight baselines for [I-daily] ...
These pairs are treated as independent fixed experiments.
No best-pair selection will be performed.

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.008001 | Val Loss: 0.003551 | beta: 1.0000 | gamma: 1.0000
Epoch 40/180 | Train Loss: 0.004381 | Val Loss: 0.001971 | beta: 1.0000 | gamma: 1.0000
Epoch 60/180 | Train Loss: 0.003227 | Val Loss: 0.001648 | beta: 1.0000 | gamma: 1.0000
Epoch 80/180 | Train Loss: 0.002726 | Val Loss: 0.001504 | beta: 1.0000 | gamma: 1.0000
Epoch 100/180 | Train Loss: 0.002399 | Val Loss: 0.001398 | beta: 1.0000 | gamma: 1.0000
Epoch 120/180 | Train Loss: 0.002291 | Val Loss: 0.001416 | beta: 1.0000 | gamma: 1.0000
Epoch 140/180 | Train Loss: 0.002344 | Val Loss: 0.001410 | beta: 1.0000 | gamma: 1.0000
Epoch 160/180 | Train Loss: 0.002203 | Val Loss: 0.001310 | beta: 1.0000 | gamma: 1.0000
Epoch 180/180 | Train Loss: 0.002175 | Val Loss: 0.001266 | beta: 1.0000 | gamma: 1.0000

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 177
 Best Val Loss: 0.001226
 beta         : 1.000000
 gamma        : 1.000000
 MAE          : 12.5651
 RMSE         : 16.9600
 MSE          : 287.6424
 R^2          : 0.9556
 Trend Acc    : 51.24%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.011285 | Val Loss: 0.006632 | beta: 0.0000 | gamma: 0.0000
Epoch 40/180 | Train Loss: 0.006953 | Val Loss: 0.004400 | beta: 0.0000 | gamma: 0.0000
Epoch 60/180 | Train Loss: 0.005277 | Val Loss: 0.003230 | beta: 0.0000 | gamma: 0.0000
Epoch 80/180 | Train Loss: 0.004253 | Val Loss: 0.002471 | beta: 0.0000 | gamma: 0.0000
Epoch 100/180 | Train Loss: 0.003579 | Val Loss: 0.002260 | beta: 0.0000 | gamma: 0.0000
Epoch 120/180 | Train Loss: 0.003120 | Val Loss: 0.002192 | beta: 0.0000 | gamma: 0.0000
Epoch 140/180 | Train Loss: 0.002794 | Val Loss: 0.001567 | beta: 0.0000 | gamma: 0.0000
Epoch 160/180 | Train Loss: 0.002583 | Val Loss: 0.001463 | beta: 0.0000 | gamma: 0.0000
Epoch 180/180 | Train Loss: 0.002421 | Val Loss: 0.001385 | beta: 0.0000 | gamma: 0.0000

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 177
 Best Val Loss: 0.001383
 beta         : 0.000000
 gamma        : 0.000000
 MAE          : 12.9433
 RMSE         : 17.3035
 MSE          : 299.4115
 R^2          : 0.9538
 Trend Acc    : 50.34%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.006009 | Val Loss: 0.003567 | beta: 0.0100 | gamma: 0.0100
Epoch 40/180 | Train Loss: 0.003660 | Val Loss: 0.002221 | beta: 0.0100 | gamma: 0.0100
Epoch 60/180 | Train Loss: 0.002733 | Val Loss: 0.001601 | beta: 0.0100 | gamma: 0.0100
Epoch 80/180 | Train Loss: 0.002419 | Val Loss: 0.001453 | beta: 0.0100 | gamma: 0.0100
Epoch 100/180 | Train Loss: 0.002174 | Val Loss: 0.001322 | beta: 0.0100 | gamma: 0.0100
Epoch 120/180 | Train Loss: 0.001959 | Val Loss: 0.001218 | beta: 0.0100 | gamma: 0.0100
Epoch 140/180 | Train Loss: 0.001919 | Val Loss: 0.001324 | beta: 0.0100 | gamma: 0.0100
Epoch 160/180 | Train Loss: 0.001831 | Val Loss: 0.001230 | beta: 0.0100 | gamma: 0.0100
Epoch 180/180 | Train Loss: 0.001827 | Val Loss: 0.001176 | beta: 0.0100 | gamma: 0.0100

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 174
 Best Val Loss: 0.001108
 beta         : 0.010000
 gamma        : 0.010000
 MAE          : 12.5791
 RMSE         : 16.7926
 MSE          : 281.9902
 R^2          : 0.9565
 Trend Acc    : 45.15%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.007008 | Val Loss: 0.004828 | beta: 0.0500 | gamma: 0.0500
Epoch 40/180 | Train Loss: 0.005343 | Val Loss: 0.002835 | beta: 0.0500 | gamma: 0.0500
Epoch 60/180 | Train Loss: 0.003490 | Val Loss: 0.001953 | beta: 0.0500 | gamma: 0.0500
Epoch 80/180 | Train Loss: 0.002960 | Val Loss: 0.001665 | beta: 0.0500 | gamma: 0.0500
Epoch 100/180 | Train Loss: 0.002538 | Val Loss: 0.001428 | beta: 0.0500 | gamma: 0.0500
Epoch 120/180 | Train Loss: 0.002291 | Val Loss: 0.001331 | beta: 0.0500 | gamma: 0.0500
Epoch 140/180 | Train Loss: 0.002079 | Val Loss: 0.001234 | beta: 0.0500 | gamma: 0.0500
Epoch 160/180 | Train Loss: 0.002004 | Val Loss: 0.001228 | beta: 0.0500 | gamma: 0.0500
Epoch 180/180 | Train Loss: 0.001960 | Val Loss: 0.001232 | beta: 0.0500 | gamma: 0.0500

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 179
 Best Val Loss: 0.001183
 beta         : 0.050000
 gamma        : 0.050000
 MAE          : 12.7000
 RMSE         : 16.9335
 MSE          : 286.7443
 R^2          : 0.9558
 Trend Acc    : 48.08%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.006433 | Val Loss: 0.003104 | beta: 0.1000 | gamma: 0.0100
Epoch 40/180 | Train Loss: 0.003755 | Val Loss: 0.002175 | beta: 0.1000 | gamma: 0.0100
Epoch 60/180 | Train Loss: 0.003398 | Val Loss: 0.002018 | beta: 0.1000 | gamma: 0.0100
Epoch 80/180 | Train Loss: 0.003055 | Val Loss: 0.001773 | beta: 0.1000 | gamma: 0.0100
Epoch 100/180 | Train Loss: 0.002746 | Val Loss: 0.001734 | beta: 0.1000 | gamma: 0.0100
Epoch 120/180 | Train Loss: 0.002373 | Val Loss: 0.001627 | beta: 0.1000 | gamma: 0.0100
Epoch 140/180 | Train Loss: 0.002231 | Val Loss: 0.001464 | beta: 0.1000 | gamma: 0.0100
Epoch 160/180 | Train Loss: 0.002165 | Val Loss: 0.001339 | beta: 0.1000 | gamma: 0.0100
Epoch 180/180 | Train Loss: 0.002097 | Val Loss: 0.001317 | beta: 0.1000 | gamma: 0.0100

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 178
 Best Val Loss: 0.001292
 beta         : 0.100000
 gamma        : 0.010000
 MAE          : 12.9670
 RMSE         : 17.1582
 MSE          : 294.4022
 R^2          : 0.9546
 Trend Acc    : 46.05%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.004860 | Val Loss: 0.002425 | beta: 0.1000 | gamma: 0.0500
Epoch 40/180 | Train Loss: 0.002822 | Val Loss: 0.001469 | beta: 0.1000 | gamma: 0.0500
Epoch 60/180 | Train Loss: 0.002337 | Val Loss: 0.001308 | beta: 0.1000 | gamma: 0.0500
Epoch 80/180 | Train Loss: 0.002299 | Val Loss: 0.001220 | beta: 0.1000 | gamma: 0.0500
Epoch 100/180 | Train Loss: 0.002101 | Val Loss: 0.001225 | beta: 0.1000 | gamma: 0.0500
Epoch 120/180 | Train Loss: 0.001951 | Val Loss: 0.001178 | beta: 0.1000 | gamma: 0.0500
Epoch 140/180 | Train Loss: 0.001879 | Val Loss: 0.001128 | beta: 0.1000 | gamma: 0.0500
Epoch 160/180 | Train Loss: 0.001840 | Val Loss: 0.001215 | beta: 0.1000 | gamma: 0.0500
Epoch 180/180 | Train Loss: 0.001947 | Val Loss: 0.001123 | beta: 0.1000 | gamma: 0.0500

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 171
 Best Val Loss: 0.001098
 beta         : 0.100000
 gamma        : 0.050000
 MAE          : 12.5998
 RMSE         : 16.7502
 MSE          : 280.5688
 R^2          : 0.9567
 Trend Acc    : 46.28%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.005843 | Val Loss: 0.004044 | beta: 0.1000 | gamma: 0.1000
Epoch 40/180 | Train Loss: 0.003255 | Val Loss: 0.001803 | beta: 0.1000 | gamma: 0.1000
Epoch 60/180 | Train Loss: 0.002747 | Val Loss: 0.001581 | beta: 0.1000 | gamma: 0.1000
Epoch 80/180 | Train Loss: 0.002329 | Val Loss: 0.001321 | beta: 0.1000 | gamma: 0.1000
Epoch 100/180 | Train Loss: 0.002284 | Val Loss: 0.001442 | beta: 0.1000 | gamma: 0.1000
Epoch 120/180 | Train Loss: 0.002050 | Val Loss: 0.001195 | beta: 0.1000 | gamma: 0.1000
Epoch 140/180 | Train Loss: 0.002047 | Val Loss: 0.001170 | beta: 0.1000 | gamma: 0.1000
Epoch 160/180 | Train Loss: 0.001998 | Val Loss: 0.001192 | beta: 0.1000 | gamma: 0.1000
Epoch 180/180 | Train Loss: 0.001893 | Val Loss: 0.001128 | beta: 0.1000 | gamma: 0.1000

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 166
 Best Val Loss: 0.001126
 beta         : 0.100000
 gamma        : 0.100000
 MAE          : 12.5759
 RMSE         : 16.7855
 MSE          : 281.7535
 R^2          : 0.9565
 Trend Acc    : 49.66%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.005799 | Val Loss: 0.003500 | beta: 0.2000 | gamma: 0.0100
Epoch 40/180 | Train Loss: 0.003957 | Val Loss: 0.002424 | beta: 0.2000 | gamma: 0.0100
Epoch 60/180 | Train Loss: 0.003264 | Val Loss: 0.002010 | beta: 0.2000 | gamma: 0.0100
Epoch 80/180 | Train Loss: 0.002877 | Val Loss: 0.001821 | beta: 0.2000 | gamma: 0.0100
Epoch 100/180 | Train Loss: 0.002533 | Val Loss: 0.001603 | beta: 0.2000 | gamma: 0.0100
Epoch 120/180 | Train Loss: 0.002400 | Val Loss: 0.001492 | beta: 0.2000 | gamma: 0.0100
Epoch 140/180 | Train Loss: 0.002303 | Val Loss: 0.001310 | beta: 0.2000 | gamma: 0.0100
Epoch 160/180 | Train Loss: 0.002032 | Val Loss: 0.001229 | beta: 0.2000 | gamma: 0.0100
Epoch 180/180 | Train Loss: 0.001853 | Val Loss: 0.001153 | beta: 0.2000 | gamma: 0.0100

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 172
 Best Val Loss: 0.001145
 beta         : 0.200000
 gamma        : 0.010000
 MAE          : 12.6224
 RMSE         : 16.8280
 MSE          : 283.1820
 R^2          : 0.9563
 Trend Acc    : 48.31%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.006248 | Val Loss: 0.004195 | beta: 0.2000 | gamma: 0.0500
Epoch 40/180 | Train Loss: 0.004446 | Val Loss: 0.003080 | beta: 0.2000 | gamma: 0.0500
Epoch 60/180 | Train Loss: 0.003143 | Val Loss: 0.002030 | beta: 0.2000 | gamma: 0.0500
Epoch 80/180 | Train Loss: 0.002870 | Val Loss: 0.001830 | beta: 0.2000 | gamma: 0.0500
Epoch 100/180 | Train Loss: 0.002361 | Val Loss: 0.001596 | beta: 0.2000 | gamma: 0.0500
Epoch 120/180 | Train Loss: 0.002195 | Val Loss: 0.001443 | beta: 0.2000 | gamma: 0.0500
Epoch 140/180 | Train Loss: 0.002096 | Val Loss: 0.001344 | beta: 0.2000 | gamma: 0.0500
Epoch 160/180 | Train Loss: 0.002008 | Val Loss: 0.001325 | beta: 0.2000 | gamma: 0.0500
Epoch 180/180 | Train Loss: 0.002065 | Val Loss: 0.001262 | beta: 0.2000 | gamma: 0.0500

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 178
 Best Val Loss: 0.001249
 beta         : 0.200000
 gamma        : 0.050000
 MAE          : 12.9245
 RMSE         : 17.1075
 MSE          : 292.6659
 R^2          : 0.9548
 Trend Acc    : 45.15%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.009738 | Val Loss: 0.005278 | beta: 0.2000 | gamma: 0.1000
Epoch 40/180 | Train Loss: 0.005407 | Val Loss: 0.002690 | beta: 0.2000 | gamma: 0.1000
Epoch 60/180 | Train Loss: 0.003822 | Val Loss: 0.002027 | beta: 0.2000 | gamma: 0.1000
Epoch 80/180 | Train Loss: 0.002932 | Val Loss: 0.001689 | beta: 0.2000 | gamma: 0.1000
Epoch 100/180 | Train Loss: 0.002778 | Val Loss: 0.001915 | beta: 0.2000 | gamma: 0.1000
Epoch 120/180 | Train Loss: 0.002394 | Val Loss: 0.001446 | beta: 0.2000 | gamma: 0.1000
Epoch 140/180 | Train Loss: 0.002257 | Val Loss: 0.001290 | beta: 0.2000 | gamma: 0.1000
Epoch 160/180 | Train Loss: 0.002084 | Val Loss: 0.001238 | beta: 0.2000 | gamma: 0.1000
Epoch 180/180 | Train Loss: 0.002172 | Val Loss: 0.001215 | beta: 0.2000 | gamma: 0.1000

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 179
 Best Val Loss: 0.001211
 beta         : 0.200000
 gamma        : 0.100000
 MAE          : 12.6721
 RMSE         : 16.9569
 MSE          : 287.5354
 R^2          : 0.9556
 Trend Acc    : 48.76%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.005356 | Val Loss: 0.003057 | beta: 0.2000 | gamma: 0.2000
Epoch 40/180 | Train Loss: 0.004532 | Val Loss: 0.002103 | beta: 0.2000 | gamma: 0.2000
Epoch 60/180 | Train Loss: 0.003082 | Val Loss: 0.001701 | beta: 0.2000 | gamma: 0.2000
Epoch 80/180 | Train Loss: 0.002518 | Val Loss: 0.001543 | beta: 0.2000 | gamma: 0.2000
Epoch 100/180 | Train Loss: 0.002440 | Val Loss: 0.001889 | beta: 0.2000 | gamma: 0.2000
Epoch 120/180 | Train Loss: 0.002397 | Val Loss: 0.001430 | beta: 0.2000 | gamma: 0.2000
Epoch 140/180 | Train Loss: 0.002199 | Val Loss: 0.001404 | beta: 0.2000 | gamma: 0.2000
Epoch 160/180 | Train Loss: 0.002102 | Val Loss: 0.001353 | beta: 0.2000 | gamma: 0.2000
Epoch 180/180 | Train Loss: 0.002138 | Val Loss: 0.001294 | beta: 0.2000 | gamma: 0.2000

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 168
 Best Val Loss: 0.001283
 beta         : 0.200000
 gamma        : 0.200000
 MAE          : 12.9498
 RMSE         : 17.1936
 MSE          : 295.6203
 R^2          : 0.9544
 Trend Acc    : 46.05%
============================================================

Starting training... [I-daily] [fixed_gridcase]
Epoch 20/180 | Train Loss: 0.007237 | Val Loss: 0.005177 | beta: 0.5000 | gamma: 0.1000
Epoch 40/180 | Train Loss: 0.004640 | Val Loss: 0.002626 | beta: 0.5000 | gamma: 0.1000
Epoch 60/180 | Train Loss: 0.003496 | Val Loss: 0.001797 | beta: 0.5000 | gamma: 0.1000
Epoch 80/180 | Train Loss: 0.002901 | Val Loss: 0.001670 | beta: 0.5000 | gamma: 0.1000
Epoch 100/180 | Train Loss: 0.002617 | Val Loss: 0.001704 | beta: 0.5000 | gamma: 0.1000
Epoch 120/180 | Train Loss: 0.002408 | Val Loss: 0.001427 | beta: 0.5000 | gamma: 0.1000
Epoch 140/180 | Train Loss: 0.002361 | Val Loss: 0.001534 | beta: 0.5000 | gamma: 0.1000
Epoch 160/180 | Train Loss: 0.002248 | Val Loss: 0.001442 | beta: 0.5000 | gamma: 0.1000
Early stopping at epoch 177 [I-daily] [fixed_gridcase]

============================================================
 I-daily | DENet (Fixed beta/gamma) [fixed_gridcase] 
============================================================
 Best Epoch   : 162
 Best Val Loss: 0.001366
 beta         : 0.500000
 gamma        : 0.100000
 MAE          : 13.0854
 RMSE         : 17.4517
 MSE          : 304.5608
 R^2          : 0.9530
 Trend Acc    : 51.47%
============================================================


Running learnable-init grid for [I-daily] ...
beta/gamma remain learnable during training; only initialization is changed.

Starting training... [I-daily] [learnable_init_grid b0=0.01, g0=0.01]
Epoch 20/180 | Train Loss: 0.004165 | Val Loss: 0.002197 | beta: -0.0823 | gamma: 0.0570
Epoch 40/180 | Train Loss: 0.003300 | Val Loss: 0.001845 | beta: -0.0880 | gamma: 0.0625
Epoch 60/180 | Train Loss: 0.002858 | Val Loss: 0.001597 | beta: -0.0924 | gamma: 0.0644
Epoch 80/180 | Train Loss: 0.002897 | Val Loss: 0.001450 | beta: -0.0933 | gamma: 0.0683
Epoch 100/180 | Train Loss: 0.002424 | Val Loss: 0.001473 | beta: -0.0957 | gamma: 0.0749
Epoch 120/180 | Train Loss: 0.002324 | Val Loss: 0.001482 | beta: -0.0946 | gamma: 0.0782
Early stopping at epoch 131 [I-daily] [learnable_init_grid b0=0.01, g0=0.01]

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.010000
 Init gamma   : 0.010000
 Best Epoch   : 116
 Best Val Loss: 0.001375
 beta         : -0.094947
 gamma        : 0.077441
 MAE          : 13.0345
 RMSE         : 17.2376
 MSE          : 297.1340
 R^2          : 0.9542
 Trend Acc    : 48.53%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.01, g0=0.02]
Epoch 20/180 | Train Loss: 0.004571 | Val Loss: 0.002225 | beta: -0.0702 | gamma: 0.1013
Epoch 40/180 | Train Loss: 0.003100 | Val Loss: 0.001651 | beta: -0.0729 | gamma: 0.1213
Epoch 60/180 | Train Loss: 0.002522 | Val Loss: 0.001489 | beta: -0.0735 | gamma: 0.1286
Epoch 80/180 | Train Loss: 0.002310 | Val Loss: 0.001335 | beta: -0.0759 | gamma: 0.1294
Epoch 100/180 | Train Loss: 0.002124 | Val Loss: 0.001351 | beta: -0.0823 | gamma: 0.1268
Epoch 120/180 | Train Loss: 0.002054 | Val Loss: 0.001240 | beta: -0.0897 | gamma: 0.1230
Epoch 140/180 | Train Loss: 0.001893 | Val Loss: 0.001166 | beta: -0.0940 | gamma: 0.1213
Epoch 160/180 | Train Loss: 0.001847 | Val Loss: 0.001216 | beta: -0.0970 | gamma: 0.1191
Epoch 180/180 | Train Loss: 0.001898 | Val Loss: 0.001143 | beta: -0.1016 | gamma: 0.1191

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.010000
 Init gamma   : 0.020000
 Best Epoch   : 178
 Best Val Loss: 0.001121
 beta         : -0.101304
 gamma        : 0.119868
 MAE          : 12.4866
 RMSE         : 16.7316
 MSE          : 279.9454
 R^2          : 0.9568
 Trend Acc    : 51.02%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.01, g0=0.05]
Epoch 20/180 | Train Loss: 0.004837 | Val Loss: 0.002709 | beta: 0.0884 | gamma: 0.1014
Epoch 40/180 | Train Loss: 0.003344 | Val Loss: 0.001925 | beta: 0.0942 | gamma: 0.1061
Epoch 60/180 | Train Loss: 0.002652 | Val Loss: 0.001682 | beta: 0.0945 | gamma: 0.1157
Epoch 80/180 | Train Loss: 0.002313 | Val Loss: 0.001458 | beta: 0.0944 | gamma: 0.1213
Epoch 100/180 | Train Loss: 0.002160 | Val Loss: 0.001321 | beta: 0.0939 | gamma: 0.1284
Epoch 120/180 | Train Loss: 0.002311 | Val Loss: 0.001750 | beta: 0.0945 | gamma: 0.1355
Epoch 140/180 | Train Loss: 0.002020 | Val Loss: 0.001274 | beta: 0.0972 | gamma: 0.1389
Epoch 160/180 | Train Loss: 0.001944 | Val Loss: 0.001597 | beta: 0.0987 | gamma: 0.1456
Epoch 180/180 | Train Loss: 0.002042 | Val Loss: 0.001155 | beta: 0.0981 | gamma: 0.1494

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.010000
 Init gamma   : 0.050000
 Best Epoch   : 173
 Best Val Loss: 0.001146
 beta         : 0.098566
 gamma        : 0.149039
 MAE          : 12.5613
 RMSE         : 16.7422
 MSE          : 280.3014
 R^2          : 0.9568
 Trend Acc    : 48.98%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.01, g0=0.1]
Epoch 20/180 | Train Loss: 0.006324 | Val Loss: 0.003364 | beta: -0.0585 | gamma: 0.1496
Epoch 40/180 | Train Loss: 0.003669 | Val Loss: 0.001960 | beta: -0.0804 | gamma: 0.1492
Epoch 60/180 | Train Loss: 0.003016 | Val Loss: 0.001669 | beta: -0.0883 | gamma: 0.1515
Epoch 80/180 | Train Loss: 0.002816 | Val Loss: 0.001565 | beta: -0.0920 | gamma: 0.1528
Epoch 100/180 | Train Loss: 0.002917 | Val Loss: 0.001398 | beta: -0.0926 | gamma: 0.1557
Epoch 120/180 | Train Loss: 0.002893 | Val Loss: 0.001494 | beta: -0.0936 | gamma: 0.1584
Epoch 140/180 | Train Loss: 0.002223 | Val Loss: 0.001296 | beta: -0.0956 | gamma: 0.1612
Epoch 160/180 | Train Loss: 0.002226 | Val Loss: 0.001293 | beta: -0.0962 | gamma: 0.1645
Epoch 180/180 | Train Loss: 0.002106 | Val Loss: 0.001242 | beta: -0.0962 | gamma: 0.1662

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.010000
 Init gamma   : 0.100000
 Best Epoch   : 175
 Best Val Loss: 0.001236
 beta         : -0.095814
 gamma        : 0.165356
 MAE          : 12.7620
 RMSE         : 16.9095
 MSE          : 285.9328
 R^2          : 0.9559
 Trend Acc    : 48.31%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.01, g0=0.2]
Epoch 20/180 | Train Loss: 0.004162 | Val Loss: 0.002435 | beta: 0.0667 | gamma: 0.1862
Epoch 40/180 | Train Loss: 0.002627 | Val Loss: 0.001553 | beta: 0.0728 | gamma: 0.1616
Epoch 60/180 | Train Loss: 0.002317 | Val Loss: 0.001280 | beta: 0.0804 | gamma: 0.1530
Epoch 80/180 | Train Loss: 0.002106 | Val Loss: 0.001241 | beta: 0.0809 | gamma: 0.1483
Epoch 100/180 | Train Loss: 0.001933 | Val Loss: 0.001252 | beta: 0.0803 | gamma: 0.1460
Early stopping at epoch 100 [I-daily] [learnable_init_grid b0=0.01, g0=0.2]

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.010000
 Init gamma   : 0.200000
 Best Epoch   : 85
 Best Val Loss: 0.001197
 beta         : 0.081950
 gamma        : 0.147752
 MAE          : 12.7489
 RMSE         : 16.9370
 MSE          : 286.8625
 R^2          : 0.9557
 Trend Acc    : 50.79%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.05, g0=0.01]
Epoch 20/180 | Train Loss: 0.004928 | Val Loss: 0.002639 | beta: 0.0505 | gamma: 0.0778
Epoch 40/180 | Train Loss: 0.003481 | Val Loss: 0.001946 | beta: 0.0415 | gamma: 0.0863
Epoch 60/180 | Train Loss: 0.002812 | Val Loss: 0.001683 | beta: 0.0426 | gamma: 0.0946
Epoch 80/180 | Train Loss: 0.002632 | Val Loss: 0.001504 | beta: 0.0497 | gamma: 0.1022
Epoch 100/180 | Train Loss: 0.002318 | Val Loss: 0.001399 | beta: 0.0588 | gamma: 0.1076
Epoch 120/180 | Train Loss: 0.002137 | Val Loss: 0.001240 | beta: 0.0764 | gamma: 0.1062
Epoch 140/180 | Train Loss: 0.001976 | Val Loss: 0.001199 | beta: 0.0875 | gamma: 0.1022
Epoch 160/180 | Train Loss: 0.001816 | Val Loss: 0.001264 | beta: 0.0990 | gamma: 0.0957
Epoch 180/180 | Train Loss: 0.002138 | Val Loss: 0.001129 | beta: 0.1115 | gamma: 0.0914

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.050000
 Init gamma   : 0.010000
 Best Epoch   : 173
 Best Val Loss: 0.001087
 beta         : 0.105714
 gamma        : 0.092912
 MAE          : 12.4568
 RMSE         : 16.6998
 MSE          : 278.8837
 R^2          : 0.9570
 Trend Acc    : 51.47%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.05, g0=0.02]
Epoch 20/180 | Train Loss: 0.006718 | Val Loss: 0.003306 | beta: 0.0305 | gamma: 0.0999
Epoch 40/180 | Train Loss: 0.004296 | Val Loss: 0.002071 | beta: 0.0437 | gamma: 0.0961
Epoch 60/180 | Train Loss: 0.003528 | Val Loss: 0.001827 | beta: 0.0486 | gamma: 0.0966
Epoch 80/180 | Train Loss: 0.002621 | Val Loss: 0.001580 | beta: 0.0568 | gamma: 0.0986
Epoch 100/180 | Train Loss: 0.002469 | Val Loss: 0.001592 | beta: 0.0613 | gamma: 0.1060
Epoch 120/180 | Train Loss: 0.002203 | Val Loss: 0.001368 | beta: 0.0641 | gamma: 0.1123
Epoch 140/180 | Train Loss: 0.002311 | Val Loss: 0.001299 | beta: 0.0676 | gamma: 0.1216
Epoch 160/180 | Train Loss: 0.002000 | Val Loss: 0.001285 | beta: 0.0709 | gamma: 0.1278
Epoch 180/180 | Train Loss: 0.001953 | Val Loss: 0.001232 | beta: 0.0721 | gamma: 0.1375

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.050000
 Init gamma   : 0.020000
 Best Epoch   : 179
 Best Val Loss: 0.001218
 beta         : 0.073446
 gamma        : 0.138395
 MAE          : 12.7208
 RMSE         : 16.9746
 MSE          : 288.1370
 R^2          : 0.9555
 Trend Acc    : 50.11%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.05, g0=0.05]
Epoch 20/180 | Train Loss: 0.006919 | Val Loss: 0.002589 | beta: 0.0819 | gamma: 0.1354
Epoch 40/180 | Train Loss: 0.004428 | Val Loss: 0.001909 | beta: 0.0894 | gamma: 0.1376
Epoch 60/180 | Train Loss: 0.003465 | Val Loss: 0.001588 | beta: 0.0946 | gamma: 0.1439
Epoch 80/180 | Train Loss: 0.002803 | Val Loss: 0.001473 | beta: 0.0991 | gamma: 0.1468
Epoch 100/180 | Train Loss: 0.002440 | Val Loss: 0.001498 | beta: 0.0990 | gamma: 0.1527
Epoch 120/180 | Train Loss: 0.002282 | Val Loss: 0.001309 | beta: 0.1006 | gamma: 0.1540
Epoch 140/180 | Train Loss: 0.002190 | Val Loss: 0.001354 | beta: 0.1006 | gamma: 0.1552
Epoch 160/180 | Train Loss: 0.002283 | Val Loss: 0.001287 | beta: 0.0996 | gamma: 0.1562
Early stopping at epoch 165 [I-daily] [learnable_init_grid b0=0.05, g0=0.05]

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.050000
 Init gamma   : 0.050000
 Best Epoch   : 150
 Best Val Loss: 0.001236
 beta         : 0.099304
 gamma        : 0.154697
 MAE          : 12.5708
 RMSE         : 16.9206
 MSE          : 286.3067
 R^2          : 0.9558
 Trend Acc    : 50.79%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.05, g0=0.1]
Epoch 20/180 | Train Loss: 0.006038 | Val Loss: 0.002680 | beta: 0.0938 | gamma: 0.1618
Epoch 40/180 | Train Loss: 0.004110 | Val Loss: 0.002119 | beta: 0.0947 | gamma: 0.1548
Epoch 60/180 | Train Loss: 0.004044 | Val Loss: 0.002068 | beta: 0.0936 | gamma: 0.1528
Epoch 80/180 | Train Loss: 0.002933 | Val Loss: 0.001533 | beta: 0.0902 | gamma: 0.1512
Epoch 100/180 | Train Loss: 0.002584 | Val Loss: 0.001454 | beta: 0.0879 | gamma: 0.1542
Epoch 120/180 | Train Loss: 0.002227 | Val Loss: 0.001388 | beta: 0.0851 | gamma: 0.1572
Epoch 140/180 | Train Loss: 0.002267 | Val Loss: 0.001277 | beta: 0.0826 | gamma: 0.1586
Epoch 160/180 | Train Loss: 0.002067 | Val Loss: 0.001306 | beta: 0.0809 | gamma: 0.1591
Epoch 180/180 | Train Loss: 0.002097 | Val Loss: 0.001332 | beta: 0.0807 | gamma: 0.1604

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.050000
 Init gamma   : 0.100000
 Best Epoch   : 169
 Best Val Loss: 0.001222
 beta         : 0.080993
 gamma        : 0.159981
 MAE          : 12.7328
 RMSE         : 16.9444
 MSE          : 287.1138
 R^2          : 0.9557
 Trend Acc    : 51.24%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.05, g0=0.2]
Epoch 20/180 | Train Loss: 0.004887 | Val Loss: 0.002648 | beta: 0.0932 | gamma: 0.2366
Epoch 40/180 | Train Loss: 0.003444 | Val Loss: 0.001828 | beta: 0.1106 | gamma: 0.2165
Epoch 60/180 | Train Loss: 0.003123 | Val Loss: 0.001777 | beta: 0.1180 | gamma: 0.2104
Epoch 80/180 | Train Loss: 0.002391 | Val Loss: 0.001516 | beta: 0.1160 | gamma: 0.2124
Epoch 100/180 | Train Loss: 0.002373 | Val Loss: 0.001433 | beta: 0.1138 | gamma: 0.2129
Epoch 120/180 | Train Loss: 0.002165 | Val Loss: 0.001373 | beta: 0.1135 | gamma: 0.2148
Epoch 140/180 | Train Loss: 0.002138 | Val Loss: 0.001323 | beta: 0.1146 | gamma: 0.2174
Epoch 160/180 | Train Loss: 0.001960 | Val Loss: 0.001226 | beta: 0.1150 | gamma: 0.2222
Epoch 180/180 | Train Loss: 0.002026 | Val Loss: 0.001175 | beta: 0.1160 | gamma: 0.2223

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.050000
 Init gamma   : 0.200000
 Best Epoch   : 180
 Best Val Loss: 0.001175
 beta         : 0.116037
 gamma        : 0.222267
 MAE          : 12.5055
 RMSE         : 16.8204
 MSE          : 282.9250
 R^2          : 0.9563
 Trend Acc    : 49.21%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.08, g0=0.01]
Epoch 20/180 | Train Loss: 0.004898 | Val Loss: 0.002310 | beta: 0.1141 | gamma: 0.0992
Epoch 40/180 | Train Loss: 0.003197 | Val Loss: 0.001755 | beta: 0.1291 | gamma: 0.0888
Epoch 60/180 | Train Loss: 0.002582 | Val Loss: 0.001585 | beta: 0.1397 | gamma: 0.0873
Epoch 80/180 | Train Loss: 0.002435 | Val Loss: 0.001434 | beta: 0.1420 | gamma: 0.0905
Epoch 100/180 | Train Loss: 0.002191 | Val Loss: 0.001328 | beta: 0.1426 | gamma: 0.0974
Epoch 120/180 | Train Loss: 0.002078 | Val Loss: 0.001254 | beta: 0.1434 | gamma: 0.1040
Epoch 140/180 | Train Loss: 0.001992 | Val Loss: 0.001196 | beta: 0.1428 | gamma: 0.1091
Epoch 160/180 | Train Loss: 0.001930 | Val Loss: 0.001170 | beta: 0.1405 | gamma: 0.1157
Epoch 180/180 | Train Loss: 0.002054 | Val Loss: 0.001217 | beta: 0.1407 | gamma: 0.1191

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.080000
 Init gamma   : 0.010000
 Best Epoch   : 176
 Best Val Loss: 0.001145
 beta         : 0.140535
 gamma        : 0.118085
 MAE          : 12.5558
 RMSE         : 16.7952
 MSE          : 282.0771
 R^2          : 0.9565
 Trend Acc    : 49.89%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.08, g0=0.02]
Epoch 20/180 | Train Loss: 0.004832 | Val Loss: 0.002205 | beta: 0.1434 | gamma: 0.1044
Epoch 40/180 | Train Loss: 0.003272 | Val Loss: 0.001504 | beta: 0.1509 | gamma: 0.1012
Epoch 60/180 | Train Loss: 0.002784 | Val Loss: 0.001426 | beta: 0.1546 | gamma: 0.1036
Epoch 80/180 | Train Loss: 0.002368 | Val Loss: 0.001349 | beta: 0.1551 | gamma: 0.1072
Epoch 100/180 | Train Loss: 0.002270 | Val Loss: 0.001327 | beta: 0.1546 | gamma: 0.1103
Epoch 120/180 | Train Loss: 0.002183 | Val Loss: 0.001189 | beta: 0.1518 | gamma: 0.1161
Epoch 140/180 | Train Loss: 0.002160 | Val Loss: 0.001243 | beta: 0.1520 | gamma: 0.1201
Epoch 160/180 | Train Loss: 0.002040 | Val Loss: 0.001172 | beta: 0.1519 | gamma: 0.1217
Early stopping at epoch 169 [I-daily] [learnable_init_grid b0=0.08, g0=0.02]

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.080000
 Init gamma   : 0.020000
 Best Epoch   : 154
 Best Val Loss: 0.001160
 beta         : 0.151847
 gamma        : 0.121749
 MAE          : 12.4755
 RMSE         : 16.7229
 MSE          : 279.6547
 R^2          : 0.9569
 Trend Acc    : 53.27%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.08, g0=0.05]
Epoch 20/180 | Train Loss: 0.008466 | Val Loss: 0.004604 | beta: 0.0494 | gamma: 0.1262
Epoch 40/180 | Train Loss: 0.004551 | Val Loss: 0.002370 | beta: 0.0846 | gamma: 0.1208
Epoch 60/180 | Train Loss: 0.003147 | Val Loss: 0.001645 | beta: 0.1161 | gamma: 0.1122
Epoch 80/180 | Train Loss: 0.002561 | Val Loss: 0.001476 | beta: 0.1332 | gamma: 0.0989
Epoch 100/180 | Train Loss: 0.002235 | Val Loss: 0.001361 | beta: 0.1387 | gamma: 0.0929
Epoch 120/180 | Train Loss: 0.002161 | Val Loss: 0.001311 | beta: 0.1397 | gamma: 0.0895
Epoch 140/180 | Train Loss: 0.002351 | Val Loss: 0.001283 | beta: 0.1406 | gamma: 0.0883
Early stopping at epoch 140 [I-daily] [learnable_init_grid b0=0.08, g0=0.05]

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.080000
 Init gamma   : 0.050000
 Best Epoch   : 125
 Best Val Loss: 0.001273
 beta         : 0.140485
 gamma        : 0.089292
 MAE          : 13.0106
 RMSE         : 17.2100
 MSE          : 296.1832
 R^2          : 0.9543
 Trend Acc    : 48.98%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.08, g0=0.1]
Epoch 20/180 | Train Loss: 0.006767 | Val Loss: 0.003635 | beta: 0.0472 | gamma: 0.1544
Epoch 40/180 | Train Loss: 0.004730 | Val Loss: 0.002179 | beta: 0.0452 | gamma: 0.1646
Epoch 60/180 | Train Loss: 0.003543 | Val Loss: 0.001898 | beta: 0.0462 | gamma: 0.1666
Epoch 80/180 | Train Loss: 0.003363 | Val Loss: 0.002004 | beta: 0.0490 | gamma: 0.1668
Epoch 100/180 | Train Loss: 0.003028 | Val Loss: 0.001615 | beta: 0.0622 | gamma: 0.1673
Epoch 120/180 | Train Loss: 0.002389 | Val Loss: 0.001423 | beta: 0.0755 | gamma: 0.1585
Epoch 140/180 | Train Loss: 0.002347 | Val Loss: 0.001411 | beta: 0.0896 | gamma: 0.1543
Epoch 160/180 | Train Loss: 0.002378 | Val Loss: 0.001246 | beta: 0.0934 | gamma: 0.1520
Epoch 180/180 | Train Loss: 0.002066 | Val Loss: 0.001177 | beta: 0.0965 | gamma: 0.1531

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.080000
 Init gamma   : 0.100000
 Best Epoch   : 177
 Best Val Loss: 0.001176
 beta         : 0.095689
 gamma        : 0.151719
 MAE          : 12.5486
 RMSE         : 16.8018
 MSE          : 282.2996
 R^2          : 0.9564
 Trend Acc    : 50.79%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.08, g0=0.2]
Epoch 20/180 | Train Loss: 0.005624 | Val Loss: 0.002741 | beta: 0.0715 | gamma: 0.2100
Epoch 40/180 | Train Loss: 0.003645 | Val Loss: 0.001970 | beta: 0.0859 | gamma: 0.2032
Epoch 60/180 | Train Loss: 0.003044 | Val Loss: 0.001659 | beta: 0.0944 | gamma: 0.1990
Epoch 80/180 | Train Loss: 0.002504 | Val Loss: 0.001391 | beta: 0.1023 | gamma: 0.2012
Epoch 100/180 | Train Loss: 0.002205 | Val Loss: 0.001271 | beta: 0.1089 | gamma: 0.1956
Epoch 120/180 | Train Loss: 0.002065 | Val Loss: 0.001285 | beta: 0.1151 | gamma: 0.1900
Epoch 140/180 | Train Loss: 0.002043 | Val Loss: 0.001169 | beta: 0.1209 | gamma: 0.1867
Epoch 160/180 | Train Loss: 0.001933 | Val Loss: 0.001208 | beta: 0.1252 | gamma: 0.1848
Epoch 180/180 | Train Loss: 0.001926 | Val Loss: 0.001141 | beta: 0.1287 | gamma: 0.1840

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.080000
 Init gamma   : 0.200000
 Best Epoch   : 177
 Best Val Loss: 0.001133
 beta         : 0.128714
 gamma        : 0.184048
 MAE          : 12.5251
 RMSE         : 16.7597
 MSE          : 280.8863
 R^2          : 0.9567
 Trend Acc    : 50.79%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.1, g0=0.01]
Epoch 20/180 | Train Loss: 0.006311 | Val Loss: 0.003782 | beta: 0.0639 | gamma: 0.0843
Epoch 40/180 | Train Loss: 0.003766 | Val Loss: 0.002271 | beta: 0.0679 | gamma: 0.0984
Epoch 60/180 | Train Loss: 0.003245 | Val Loss: 0.001608 | beta: 0.0671 | gamma: 0.1047
Epoch 80/180 | Train Loss: 0.003196 | Val Loss: 0.001520 | beta: 0.0655 | gamma: 0.1125
Epoch 100/180 | Train Loss: 0.002470 | Val Loss: 0.001383 | beta: 0.0635 | gamma: 0.1186
Epoch 120/180 | Train Loss: 0.002303 | Val Loss: 0.001365 | beta: 0.0626 | gamma: 0.1214
Epoch 140/180 | Train Loss: 0.002268 | Val Loss: 0.001281 | beta: 0.0631 | gamma: 0.1251
Epoch 160/180 | Train Loss: 0.002270 | Val Loss: 0.001295 | beta: 0.0642 | gamma: 0.1260
Epoch 180/180 | Train Loss: 0.002187 | Val Loss: 0.001307 | beta: 0.0656 | gamma: 0.1284

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.100000
 Init gamma   : 0.010000
 Best Epoch   : 175
 Best Val Loss: 0.001229
 beta         : 0.065155
 gamma        : 0.128824
 MAE          : 12.6602
 RMSE         : 16.9055
 MSE          : 285.7951
 R^2          : 0.9559
 Trend Acc    : 48.98%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.1, g0=0.02]
Epoch 20/180 | Train Loss: 0.004070 | Val Loss: 0.002038 | beta: 0.0764 | gamma: 0.1234
Epoch 40/180 | Train Loss: 0.002745 | Val Loss: 0.001609 | beta: 0.0848 | gamma: 0.1109
Epoch 60/180 | Train Loss: 0.002332 | Val Loss: 0.001416 | beta: 0.0878 | gamma: 0.1048
Epoch 80/180 | Train Loss: 0.002069 | Val Loss: 0.001285 | beta: 0.0919 | gamma: 0.1021
Epoch 100/180 | Train Loss: 0.002130 | Val Loss: 0.001328 | beta: 0.0920 | gamma: 0.1035
Epoch 120/180 | Train Loss: 0.001940 | Val Loss: 0.001158 | beta: 0.0951 | gamma: 0.1078
Epoch 140/180 | Train Loss: 0.002011 | Val Loss: 0.001157 | beta: 0.0991 | gamma: 0.1107
Epoch 160/180 | Train Loss: 0.001769 | Val Loss: 0.001125 | beta: 0.1006 | gamma: 0.1142
Epoch 180/180 | Train Loss: 0.001869 | Val Loss: 0.001111 | beta: 0.1011 | gamma: 0.1176

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.100000
 Init gamma   : 0.020000
 Best Epoch   : 179
 Best Val Loss: 0.001090
 beta         : 0.101036
 gamma        : 0.116777
 MAE          : 12.5291
 RMSE         : 16.7529
 MSE          : 280.6606
 R^2          : 0.9567
 Trend Acc    : 47.40%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.1, g0=0.05]
Epoch 20/180 | Train Loss: 0.006029 | Val Loss: 0.002514 | beta: 0.0921 | gamma: 0.1301
Epoch 40/180 | Train Loss: 0.003677 | Val Loss: 0.001746 | beta: 0.1373 | gamma: 0.1335
Epoch 60/180 | Train Loss: 0.003571 | Val Loss: 0.001494 | beta: 0.1683 | gamma: 0.1347
Epoch 80/180 | Train Loss: 0.002625 | Val Loss: 0.001574 | beta: 0.1763 | gamma: 0.1329
Early stopping at epoch 99 [I-daily] [learnable_init_grid b0=0.1, g0=0.05]

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.100000
 Init gamma   : 0.050000
 Best Epoch   : 84
 Best Val Loss: 0.001360
 beta         : 0.178444
 gamma        : 0.134878
 MAE          : 13.0245
 RMSE         : 17.3605
 MSE          : 301.3875
 R^2          : 0.9535
 Trend Acc    : 51.69%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.1, g0=0.1]
Epoch 20/180 | Train Loss: 0.006010 | Val Loss: 0.003386 | beta: 0.0806 | gamma: 0.1475
Epoch 40/180 | Train Loss: 0.003401 | Val Loss: 0.002172 | beta: 0.0649 | gamma: 0.1604
Epoch 60/180 | Train Loss: 0.002846 | Val Loss: 0.001742 | beta: 0.0576 | gamma: 0.1735
Epoch 80/180 | Train Loss: 0.002813 | Val Loss: 0.001770 | beta: 0.0539 | gamma: 0.1784
Epoch 100/180 | Train Loss: 0.002423 | Val Loss: 0.001560 | beta: 0.0558 | gamma: 0.1829
Epoch 120/180 | Train Loss: 0.002434 | Val Loss: 0.001383 | beta: 0.0539 | gamma: 0.1893
Epoch 140/180 | Train Loss: 0.002102 | Val Loss: 0.001270 | beta: 0.0588 | gamma: 0.1917
Epoch 160/180 | Train Loss: 0.002055 | Val Loss: 0.001314 | beta: 0.0665 | gamma: 0.1930
Epoch 180/180 | Train Loss: 0.001971 | Val Loss: 0.001190 | beta: 0.0698 | gamma: 0.1954

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.100000
 Init gamma   : 0.100000
 Best Epoch   : 175
 Best Val Loss: 0.001186
 beta         : 0.070402
 gamma        : 0.192490
 MAE          : 12.5302
 RMSE         : 16.8379
 MSE          : 283.5144
 R^2          : 0.9563
 Trend Acc    : 50.34%
============================================================

Starting training... [I-daily] [learnable_init_grid b0=0.1, g0=0.2]
Epoch 20/180 | Train Loss: 0.006940 | Val Loss: 0.003209 | beta: 0.1233 | gamma: 0.2096
Epoch 40/180 | Train Loss: 0.003698 | Val Loss: 0.001945 | beta: 0.1383 | gamma: 0.1942
Epoch 60/180 | Train Loss: 0.002686 | Val Loss: 0.001671 | beta: 0.1469 | gamma: 0.1870
Epoch 80/180 | Train Loss: 0.002403 | Val Loss: 0.001610 | beta: 0.1506 | gamma: 0.1818
Epoch 100/180 | Train Loss: 0.003236 | Val Loss: 0.001368 | beta: 0.1525 | gamma: 0.1777
Epoch 120/180 | Train Loss: 0.002107 | Val Loss: 0.001235 | beta: 0.1517 | gamma: 0.1751
Epoch 140/180 | Train Loss: 0.002583 | Val Loss: 0.001241 | beta: 0.1519 | gamma: 0.1730
Epoch 160/180 | Train Loss: 0.001984 | Val Loss: 0.001197 | beta: 0.1507 | gamma: 0.1714
Epoch 180/180 | Train Loss: 0.002073 | Val Loss: 0.001201 | beta: 0.1517 | gamma: 0.1726

============================================================
 I-daily | DENet (Learnable init-grid) 
============================================================
 Init beta    : 0.100000
 Init gamma   : 0.200000
 Best Epoch   : 179
 Best Val Loss: 0.001131
 beta         : 0.151246
 gamma        : 0.172426
 MAE          : 12.5100
 RMSE         : 16.7186
 MSE          : 279.5107
 R^2          : 0.9569
 Trend Acc    : 50.56%
============================================================


Ablation summary saved to: ../daily_ablation/ablation_fusion_weights_summary.csv

========================================================================================================================
 Ablation Summary 
========================================================================================================================
asset_name frequency            model_variant  init_beta  init_gamma      beta    gamma  ATR_mean  roll_return_std       MAE      RMSE       R2  Trend_Acc
         I     daily          DENet_learnable       0.20        0.01  0.116179 0.128128   0.02875         0.006949 12.427663 16.605381 0.957455  50.112867
         I     daily     DENet_fixed_gridcase        NaN         NaN  1.000000 1.000000   0.02875         0.006949 12.565126 16.960023 0.955619  51.241535
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.000000 0.000000   0.02875         0.006949 12.943303 17.303511 0.953803  50.338600
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.010000 0.010000   0.02875         0.006949 12.579142 16.792564 0.956491  45.146727
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.050000 0.050000   0.02875         0.006949 12.700046 16.933527 0.955757  48.081264
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.100000 0.010000   0.02875         0.006949 12.967005 17.158154 0.954576  46.049661
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.100000 0.050000   0.02875         0.006949 12.599790 16.750187 0.956710  46.275395
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.100000 0.100000   0.02875         0.006949 12.575922 16.785516 0.956527  49.661400
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.200000 0.010000   0.02875         0.006949 12.622447 16.828012 0.956307  48.306998
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.200000 0.050000   0.02875         0.006949 12.924468 17.107482 0.954844  45.146727
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.200000 0.100000   0.02875         0.006949 12.672123 16.956868 0.955635  48.758465
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.200000 0.200000   0.02875         0.006949 12.949769 17.193613 0.954388  46.049661
         I     daily     DENet_fixed_gridcase        NaN         NaN  0.500000 0.100000   0.02875         0.006949 13.085362 17.451670 0.953008  51.467269
         I     daily DENet_learnable_initgrid       0.01        0.01 -0.094947 0.077441   0.02875         0.006949 13.034518 17.237575 0.954154  48.532731
         I     daily DENet_learnable_initgrid       0.01        0.02 -0.101304 0.119868   0.02875         0.006949 12.486584 16.731570 0.956806  51.015801
         I     daily DENet_learnable_initgrid       0.01        0.05  0.098566 0.149039   0.02875         0.006949 12.561313 16.742203 0.956752  48.984199
         I     daily DENet_learnable_initgrid       0.01        0.10 -0.095814 0.165356   0.02875         0.006949 12.761995 16.909548 0.955883  48.306998
         I     daily DENet_learnable_initgrid       0.01        0.20  0.081950 0.147752   0.02875         0.006949 12.748922 16.937016 0.955739  50.790068
         I     daily DENet_learnable_initgrid       0.05        0.01  0.105714 0.092912   0.02875         0.006949 12.456763 16.699812 0.956970  51.467269
         I     daily DENet_learnable_initgrid       0.05        0.02  0.073446 0.138395   0.02875         0.006949 12.720772 16.974598 0.955543  50.112867
         I     daily DENet_learnable_initgrid       0.05        0.05  0.099304 0.154697   0.02875         0.006949 12.570775 16.920599 0.955825  50.790068
         I     daily DENet_learnable_initgrid       0.05        0.10  0.080993 0.159981   0.02875         0.006949 12.732831 16.944432 0.955700  51.241535
         I     daily DENet_learnable_initgrid       0.05        0.20  0.116037 0.222267   0.02875         0.006949 12.505467 16.820375 0.956347  49.209932
         I     daily DENet_learnable_initgrid       0.08        0.01  0.140535 0.118085   0.02875         0.006949 12.555804 16.795151 0.956478  49.887133
         I     daily DENet_learnable_initgrid       0.08        0.02  0.151847 0.121749   0.02875         0.006949 12.475506 16.722880 0.956851  53.273138
         I     daily DENet_learnable_initgrid       0.08        0.05  0.140485 0.089292   0.02875         0.006949 13.010623 17.209974 0.954301  48.984199
         I     daily DENet_learnable_initgrid       0.08        0.10  0.095689 0.151719   0.02875         0.006949 12.548565 16.801775 0.956443  50.790068
         I     daily DENet_learnable_initgrid       0.08        0.20  0.128714 0.184048   0.02875         0.006949 12.525064 16.759662 0.956661  50.790068
         I     daily DENet_learnable_initgrid       0.10        0.01  0.065155 0.128824   0.02875         0.006949 12.660214 16.905475 0.955904  48.984199
         I     daily DENet_learnable_initgrid       0.10        0.02  0.101036 0.116777   0.02875         0.006949 12.529095 16.752927 0.956696  47.404063
         I     daily DENet_learnable_initgrid       0.10        0.05  0.178444 0.134878   0.02875         0.006949 13.024518 17.360514 0.953498  51.693002
         I     daily DENet_learnable_initgrid       0.10        0.10  0.070402 0.192490   0.02875         0.006949 12.530165 16.837885 0.956256  50.338600
         I     daily DENet_learnable_initgrid       0.10        0.20  0.151246 0.172426   0.02875         0.006949 12.510000 16.718573 0.956874  50.564334
========================================================================================================================

========================================================================================================================
 Learned Fusion Weights (Default init) 
========================================================================================================================
asset_name frequency  init_beta  init_gamma     beta    gamma  ATR_mean  roll_return_std       MAE  Trend_Acc
         I     daily        0.2        0.01 0.116179 0.128128   0.02875         0.006949 12.427663  50.112867
========================================================================================================================

========================================================================================================================
 Learnable Init Grid Summary 
========================================================================================================================
asset_name frequency  init_beta  init_gamma      beta    gamma       MAE      RMSE       R2  Trend_Acc
         I     daily       0.05        0.01  0.105714 0.092912 12.456763 16.699812 0.956970  51.467269
         I     daily       0.08        0.02  0.151847 0.121749 12.475506 16.722880 0.956851  53.273138
         I     daily       0.01        0.02 -0.101304 0.119868 12.486584 16.731570 0.956806  51.015801
         I     daily       0.05        0.20  0.116037 0.222267 12.505467 16.820375 0.956347  49.209932
         I     daily       0.10        0.20  0.151246 0.172426 12.510000 16.718573 0.956874  50.564334
         I     daily       0.08        0.20  0.128714 0.184048 12.525064 16.759662 0.956661  50.790068
         I     daily       0.10        0.02  0.101036 0.116777 12.529095 16.752927 0.956696  47.404063
         I     daily       0.10        0.10  0.070402 0.192490 12.530165 16.837885 0.956256  50.338600
         I     daily       0.08        0.10  0.095689 0.151719 12.548565 16.801775 0.956443  50.790068
         I     daily       0.08        0.01  0.140535 0.118085 12.555804 16.795151 0.956478  49.887133
         I     daily       0.01        0.05  0.098566 0.149039 12.561313 16.742203 0.956752  48.984199
         I     daily       0.05        0.05  0.099304 0.154697 12.570775 16.920599 0.955825  50.790068
         I     daily       0.10        0.01  0.065155 0.128824 12.660214 16.905475 0.955904  48.984199
         I     daily       0.05        0.02  0.073446 0.138395 12.720772 16.974598 0.955543  50.112867
         I     daily       0.05        0.10  0.080993 0.159981 12.732831 16.944432 0.955700  51.241535
         I     daily       0.01        0.20  0.081950 0.147752 12.748922 16.937016 0.955739  50.790068
         I     daily       0.01        0.10 -0.095814 0.165356 12.761995 16.909548 0.955883  48.306998
         I     daily       0.08        0.05  0.140485 0.089292 13.010623 17.209974 0.954301  48.984199
         I     daily       0.10        0.05  0.178444 0.134878 13.024518 17.360514 0.953498  51.693002
         I     daily       0.01        0.01 -0.094947 0.077441 13.034518 17.237575 0.954154  48.532731
========================================================================================================================

========================================================================================================================
 Fixed-Weight Cases Summary 
========================================================================================================================
asset_name frequency  beta  gamma       MAE      RMSE       R2  Trend_Acc
         I     daily  1.00   1.00 12.565126 16.960023 0.955619  51.241535
         I     daily  0.00   0.00 12.943303 17.303511 0.953803  50.338600
         I     daily  0.01   0.01 12.579142 16.792564 0.956491  45.146727
         I     daily  0.05   0.05 12.700046 16.933527 0.955757  48.081264
         I     daily  0.10   0.01 12.967005 17.158154 0.954576  46.049661
         I     daily  0.10   0.05 12.599790 16.750187 0.956710  46.275395
         I     daily  0.10   0.10 12.575922 16.785516 0.956527  49.661400
         I     daily  0.20   0.01 12.622447 16.828012 0.956307  48.306998
         I     daily  0.20   0.05 12.924468 17.107482 0.954844  45.146727
         I     daily  0.20   0.10 12.672123 16.956868 0.955635  48.758465
         I     daily  0.20   0.20 12.949769 17.193613 0.954388  46.049661
         I     daily  0.50   0.10 13.085362 17.451670 0.953008  51.467269
========================================================================================================================

进程已结束，退出代码为 0

"""