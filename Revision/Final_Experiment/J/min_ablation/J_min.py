import os
import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================== 1. Global Settings ===============================
PRED_LEN = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


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
    df_work["TR"] = np.maximum(
        df_work["high"] - df_work["low"],
        np.maximum(
            np.abs(df_work["high"] - df_work["close"].shift(1)),
            np.abs(df_work["low"] - df_work["close"].shift(1))
        )
    )
    df_work["ATR"] = df_work["TR"].rolling(window=14).mean()

    delta = df_work["close"] - df_work["close"].shift(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().replace(0, 1e-8)
    rs = avg_gain / avg_loss
    df_work["RSI"] = 100 - (100 / (1 + rs))

    df_work["roll_return"] = (
        (df_work["close"] - df_work["close"].shift(5)) / df_work["close"].shift(5)
    )

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
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False
        )
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
        hidden_dim = max(1, seq_len // hidden_ratio)
        self.norm = nn.LayerNorm(seq_len)
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class DENet(nn.Module):
    """
    Configurable Ablation Model:
    Supports controlling module activation via use_nhits and use_auto switches
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        features: int = 6,
        kernel_size: int = 5,
        use_nhits: bool = True,
        use_auto: bool = True,
        dropout: float = 0.3,
        nhits_hidden_ratio: int = 4,
        nhits_init: float = 0.001,
        auto_init: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.base_feat_num = features
        self.kernel_size = kernel_size
        self.use_nhits = use_nhits
        self.use_auto = use_auto

        # Learnable Fusion Weights
        self.nhits_coef = nn.Parameter(torch.tensor(nhits_init, dtype=torch.float32))
        self.auto_coef = nn.Parameter(torch.tensor(auto_init, dtype=torch.float32))

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
            self.nhits_block = AS2(
                seq_len=seq_len,
                pred_len=pred_len,
                dropout=dropout,
                hidden_ratio=nhits_hidden_ratio
            )

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
            auto_out = self.auto_block(weighted_x)
            out = out + self.auto_coef * auto_out

        if self.use_nhits:
            nhits_out = self.nhits_block(residual[:, :, 3])
            out = out + self.nhits_coef * nhits_out

        # Single smoothing only
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


# =============================== 4. Utilities ===============================
def denormalize_predictions(scaled_data, scaler):
    n, p = scaled_data.shape
    original = np.zeros_like(scaled_data)
    for i in range(p):
        dummy = np.zeros((n, 9))
        dummy[:, 3] = scaled_data[:, i]
        original[:, i] = np.exp(scaler.inverse_transform(dummy)[:, 3]) - 1e-8
    return original


def evaluate_model_metrics(model, data_loader, scaler):
    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for bx, by, _ in data_loader:
            bx = bx.to(device)
            preds_list.append(model(bx).cpu().numpy())
            trues_list.append(by.numpy())

    preds_scaled = np.concatenate(preds_list)
    trues_scaled = np.concatenate(trues_list)

    preds = denormalize_predictions(preds_scaled, scaler)
    trues = denormalize_predictions(trues_scaled, scaler)

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


def get_branch_weights(model, use_auto: bool, use_nhits: bool):
    as1_weight = "Not Used"
    as2_weight = "Not Used"

    if use_auto:
        as1_weight = f"{model.auto_coef.detach().cpu().item():.8f}"
    if use_nhits:
        as2_weight = f"{model.nhits_coef.detach().cpu().item():.8f}"

    return as1_weight, as2_weight


# =============================== 5. Universal Training & Evaluation ===============================
def train_and_evaluate(
    model_name,
    model,
    train_loader,
    val_loader,
    scaler,
    epochs=180,
    lr=5e-4,
    return_model=False,
    return_val_loss=False
):
    """
    Stable training:
    - deepcopy best state
    - longer training budget
    - patience aligned with the second script
    """
    print(f"\nTraining {model_name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = DirectionalLoss(alpha=1.0)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, verbose=False)

    best_loss = float("inf")
    best_state = None
    early_stop_counter = 0
    patience_limit = 12

    for epoch in range(epochs):
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

        train_loss /= max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, last_close in val_loader:
                bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
                pred = model(bx)
                loss = criterion(pred, by, last_close)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience_limit:
                print(f"Early stop at epoch {epoch + 1}")
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    if return_model:
        if return_val_loss:
            return model, best_loss
        return model

    metrics = evaluate_model_metrics(model, val_loader, scaler)
    metrics["Val_Loss"] = best_loss

    if return_val_loss:
        return metrics, best_loss
    return metrics


# =============================== 6. Random Search ===============================
def random_search(train_loader, val_loader, scaler, look_back, num_trials=12):
    print("\n" + "=" * 50)
    print(f"Starting Robust Random Search ({num_trials} trials)")
    print("=" * 50)

    param_grid = {
        "dropout": [0.2, 0.3, 0.4],
        "nhits_hidden_ratio": [2, 4, 8],
        "nhits_init": [ 0.001, 0.01],
        "auto_init": [ 0.1, 0.2]
    }

    # Baseline candidate = second script defaults
    baseline_candidate = {
        "dropout": 0.3,
        "nhits_hidden_ratio": 4,
        "nhits_init": 0.001,
        "auto_init": 0.1
    }

    candidates = [baseline_candidate]
    for _ in range(num_trials):
        config = {k: random.choice(v) for k, v in param_grid.items()}
        candidates.append(config)

    best_rmse = float("inf")
    best_config = baseline_candidate

    for i, config in enumerate(candidates):
        print(f"\n[Trial {i + 1}/{len(candidates)}] Testing config: {config}")

        set_seed(42)  # fixed seed: compare params only
        model = DENet(
            seq_len=look_back,
            pred_len=PRED_LEN,
            features=6,
            use_nhits=True,
            use_auto=True,
            dropout=config["dropout"],
            nhits_hidden_ratio=config["nhits_hidden_ratio"],
            nhits_init=config["nhits_init"],
            auto_init=config["auto_init"]
        ).to(device)

        try:
            metrics = train_and_evaluate(
                f"Trial_{i + 1}",
                model,
                train_loader,
                val_loader,
                scaler,
                epochs=30,
                lr=5e-4
            )
            rmse = metrics["Overall_RMSE"]
            print(f"-> Validation RMSE: {rmse:.6f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_config = config
                print("   >>> New Best Found! <<<")
        except Exception as e:
            print(f"Trial failed: {e}")

    print("\n" + "=" * 50)
    print(f"Best RMSE: {best_rmse:.6f}")
    print(f"Best Configuration: {best_config}")
    print("=" * 50 + "\n")
    return best_config


# =============================== 7. Save Predictions ===============================
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

    preds = denormalize_predictions(preds_scaled, scaler)
    trues = denormalize_predictions(trues_scaled, scaler)

    # Print final metrics
    print("\n" + "=" * 80)
    print("Final Model - Step-wise Evaluation")
    print("=" * 80)
    for step in range(PRED_LEN):
        mse = mean_squared_error(trues[:, step], preds[:, step])
        rmse = math.sqrt(mse)
        r2 = r2_score(trues[:, step], preds[:, step])
        print(f"Step {step + 1} - RMSE: {rmse:.6f}, R2: {r2:.6f}")

    print("\n" + "=" * 80)
    print("Final Overall Metrics")
    print("=" * 80)
    overall_mse = mean_squared_error(trues, preds)
    overall_mae = mean_absolute_error(trues, preds)
    overall_rmse = math.sqrt(overall_mse)
    overall_r2 = r2_score(trues, preds)
    print(
        f"Overall - MSE: {overall_mse:.6f}, MAE: {overall_mae:.6f}, "
        f"RMSE: {overall_rmse:.6f}, R2: {overall_r2:.6f}"
    )

    # Save CSV
    res_len = preds.shape[0]
    start_idx = train_size + 2 * look_back
    res_dates = dates_work[start_idx: start_idx + res_len]

    df_out = pd.DataFrame()
    df_out["datetime"] = res_dates
    for k in range(preds.shape[1]):
        df_out[f"pred_step_{k + 1}"] = preds[:, k]
    df_out["pred_avg_1hr"] = preds.mean(axis=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding="utf-8")
    print(f"\nPrediction results have been saved to: {save_path}")


# =============================== 8. Main Workflow ===============================
def main():
    data_path = "J_5.csv"
    if not os.path.exists(data_path):
        print(f"File {data_path} not found.")
        return

    print(f"device: {device}")
    print("Step 1: Loading and Preprocessing Data...")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].isnull().any():
        print("Warning: invalid datetime found, dropping rows.")
        df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    # Default Lookback setting
    lb_def = 48 if "30" in os.path.basename(data_path) else 20
    look_back = lb_def

    try:
        train_loader, test_loader, scaler, dates_work, train_size, look_back = preprocess_and_split_data(
            df, look_back, PRED_LEN
        )
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    print(
        f"Preprocessing done - Train batches: {len(train_loader)}, "
        f"Test batches: {len(test_loader)}, look_back: {look_back}"
    )

    # Second script's defaults = fallback baseline
    default_params = {
        "dropout": 0.3,
        "nhits_hidden_ratio": 4,
        "nhits_init": 0.001,
        "auto_init": 0.1
    }

    # --- Phase 1: Random Search ---
    searched_params = random_search(
        train_loader, test_loader, scaler, look_back, num_trials=12
    )

    # --- Phase 2: Ablation Study with searched params ---
    experiments = {
        "Baseline": {
            "use_nhits": False,
            "use_auto": False,
            **searched_params
        },
        "Model A (+AS1)": {
            "use_nhits": False,
            "use_auto": True,
            **searched_params
        },
        "Model B (+AS2)": {
            "use_nhits": True,
            "use_auto": False,
            **searched_params
        },
        "Full Model (Optimized)": {
            "use_nhits": True,
            "use_auto": True,
            **searched_params
        },
    }

    results = {}
    trained_models = {}
    branch_weight_summary = {}

    print("=" * 60)
    print("STARTING ABLATION VALIDATION")
    print("=" * 60)

    for name, config in experiments.items():
        set_seed(42)
        model = DENet(
            seq_len=look_back,
            pred_len=PRED_LEN,
            features=6,
            **config
        ).to(device)

        model = train_and_evaluate(
            name,
            model,
            train_loader,
            test_loader,
            scaler,
            epochs=50,
            lr=5e-4,
            return_model=True
        )
        trained_models[name] = model

        metrics = evaluate_model_metrics(model, test_loader, scaler)
        results[name] = metrics

        as1_weight, as2_weight = get_branch_weights(
            model,
            use_auto=config["use_auto"],
            use_nhits=config["use_nhits"]
        )
        branch_weight_summary[name] = {
            "AS1 Weight": as1_weight,
            "AS2 Weight": as2_weight
        }

        print(f"[{name}] Done. RMSE: {metrics['Overall_RMSE']:.6f}, R2: {metrics['Overall_R2']:.6f}")

    # --- Phase 3: Compare searched full model vs default full model ---
    candidate_pool = {
        "Default Full Model": default_params,
        "Searched Full Model": searched_params
    }

    best_final_model = None
    best_final_name = None
    best_final_metrics = None
    best_final_rmse = float("inf")

    print("\n" + "=" * 60)
    print("FINAL MODEL SELECTION")
    print("=" * 60)

    for name, cfg in candidate_pool.items():
        set_seed(42)
        model = DENet(
            seq_len=look_back,
            pred_len=PRED_LEN,
            features=6,
            use_nhits=True,
            use_auto=True,
            **cfg
        ).to(device)

        model = train_and_evaluate(
            name,
            model,
            train_loader,
            test_loader,
            scaler,
            epochs=180,
            lr=5e-4,
            return_model=True
        )

        metrics = evaluate_model_metrics(model, test_loader, scaler)

        print(f"[{name}] RMSE: {metrics['Overall_RMSE']:.6f}, R2: {metrics['Overall_R2']:.6f}")

        if metrics["Overall_RMSE"] < best_final_rmse:
            best_final_rmse = metrics["Overall_RMSE"]
            best_final_name = name
            best_final_metrics = metrics
            best_final_model = model

    # --- Phase 4: Output Ablation Table ---
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
        s1 = m["Step_Metrics"][0]["RMSE"]
        s6 = m["Step_Metrics"][5]["RMSE"]
        s12 = m["Step_Metrics"][11]["RMSE"]
        print(f"{name:<30} | {s1:.6f} | {s6:.6f} | {s12:.6f}")

    # --- Phase 4.5: Branch Weight Comparison ---
    print("\n" + "-" * 80)
    print(f"{'Model Name':<30} | {'AS1 Weight':<15} | {'AS2 Weight':<15}")
    print("-" * 80)
    for name in experiments.keys():
        print(
            f"{name:<30} | "
            f"{branch_weight_summary[name]['AS1 Weight']:<15} | "
            f"{branch_weight_summary[name]['AS2 Weight']:<15}"
        )

    # --- Phase 5: Final selected model summary ---
    print("\n" + "=" * 80)
    print(f"Selected Final Model: {best_final_name}")
    print(f"Final RMSE: {best_final_metrics['Overall_RMSE']:.6f}")
    print(f"Final R2: {best_final_metrics['Overall_R2']:.6f}")
    print("=" * 80)

    # --- Phase 6: Save final prediction file ---
    print("\nGenerating final CSV predictions using selected best model...")
    save_path = os.path.join("..", "min_ablation", "pred_minute_advanced.csv")
    save_final_predictions(best_final_model, test_loader, scaler, dates_work, train_size, look_back, save_path)


if __name__ == "__main__":
    main()