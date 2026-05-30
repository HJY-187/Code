import os
import math
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

PRED_LEN = 12


# =============================== data ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        # x: (open/high/low/close/volume/open_interest)
        x = self.data[idx:idx + self.seq_len, :6]
        # y: close
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, 3]
        current_price = self.data[idx + self.seq_len - 1, 3]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(current_price, dtype=torch.float32),
        )


# =============================== Data Preprocessing and Splitting ===============================
def preprocess_and_split_data(df: pd.DataFrame, look_back: int, pred_len: int):
    df_work = df.copy()

    # log()
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # ATR
    df_work["TR"] = np.maximum(
        df_work["high"] - df_work["low"],
        np.maximum(
            np.abs(df_work["high"] - df_work["close"].shift(1)),
            np.abs(df_work["low"] - df_work["close"].shift(1))
        )
    )
    df_work["ATR"] = df_work["TR"].rolling(window=14).mean()

    # RSI
    delta = df_work["close"] - df_work["close"].shift(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().replace(0, 1e-8)
    rs = avg_gain / avg_loss
    df_work["RSI"] = 100 - (100 / (1 + rs))

    # roll_return
    df_work["roll_return"] = (
        (df_work["close"] - df_work["close"].shift(5)) / df_work["close"].shift(5)
    )

    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # features
    features = df_work[
        [
            "open", "high", "low", "close", "volume", "open_interest",
            "ATR", "RSI", "roll_return"
        ]
    ].values

    # dates
    dates_work = df_work["datetime"].values

    # 8:2 split with look_back gap
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


# =============================== model ===============================
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

    def forward(self, x):  # [B,C,S]
        y = self.depthwise(x)
        y = F.gelu(y)
        y = self.pointwise(y)
        y = self.bn(y)
        return y  # [B,1,S]


class AuxiliaryStreams1(nn.Module):
    def __init__(self, channels: int, seq_len: int, pred_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.conv = DepthwiseSeparableConv1d(channels, kernel_size=7, dilation=2)
        self.linear = nn.Linear(seq_len, pred_len)

    def moving_avg(self, x, kernel_size=5):
        padding = (kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):  # x: [B,S,F]
        trend = self.moving_avg(x, kernel_size=7)
        seasonal = x - trend
        y = self.conv(seasonal.permute(0, 2, 1))  # [B,1,S]
        y = y.squeeze(1)  # [B,S]
        out = self.linear(y)  # [B,pred_len]
        return out


class AuxiliaryStreams2(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, dropout: float = 0.3, hidden_ratio: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        hidden_dim = max(1, seq_len // hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )

    def forward(self, x):  # x: [B,S] (close)
        return self.net(self.norm(x))


# =============================== Main Path ===============================
class DENet(nn.Module):
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
        nhits_coef: float = 0.001,   # fixed weight from grid search
        auto_coef: float = 0.1       # fixed weight from grid search
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.base_feat_num = features
        self.kernel_size = kernel_size
        self.use_nhits = use_nhits
        self.use_auto = use_auto

        # fixed fusion weights
        self.nhits_coef = nhits_coef
        self.auto_coef = auto_coef

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
            self.auto_block = AuxiliaryStreams1(
                channels=features, seq_len=seq_len, pred_len=pred_len
            )

        if self.use_nhits:
            self.nhits_block = AuxiliaryStreams2(
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

    def forward(self, x):  # [B, seq_len, features]
        x_base = x[:, :, :self.base_feat_num]
        w = self.feature_weights.unsqueeze(0).unsqueeze(0)
        weighted_x = x_base * w

        trend = self.moving_avg(weighted_x, self.kernel_size)
        residual = weighted_x - trend

        trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_linear(residual.permute(0, 2, 1)).permute(0, 2, 1)

        base_pred = (trend_pred + residual_pred)[:, :, 3]  # close
        base_pred = base_pred + 0.1 * self.decoder(base_pred)

        out = base_pred

        if self.use_auto:
            auto_out = self.auto_block(weighted_x)
            out = out + self.auto_coef * auto_out

        if self.use_nhits:
            nhits_out = self.nhits_block(residual[:, :, 3])
            out = out + self.nhits_coef * nhits_out

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

        # sign mismatch penalty
        penalty = F.relu(-1.0 * true_diff * pred_diff)

        return num_loss + self.alpha * torch.mean(penalty)


def denormalize_predictions(scaled_preds, scaler):
    """Inverse scaling for close column"""
    n, p = scaled_preds.shape
    original = np.zeros_like(scaled_preds)
    for i in range(p):
        dummy = np.zeros((n, 9))
        dummy[:, 3] = scaled_preds[:, i]
        den = scaler.inverse_transform(dummy)
        original[:, i] = np.exp(den[:, 3]) - 1e-8
    return original


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=8e-4,
    model_dir="models_DENet",
    model_name="DENet_best.pth"
):
    """Early stopping + LR scheduler"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    crit = DirectionalLoss(alpha=1.0)

    best_val_loss = float("inf")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, model_name)
    patience = 0
    early_stop_patience = 12

    for ep in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            last_close = bx[:, -1, 3]

            opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, by, last_close)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item()

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                last_close = bx[:, -1, 3]
                pred = model(bx)
                val_loss += crit(pred, by, last_close).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        sch.step(val_loss)

        if ep % 20 == 0:
            print(
                f"Epoch {ep + 1}/{epochs} - "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(
                    f"Early stop at epoch {ep + 1} "
                    f"(val loss no improvement for {early_stop_patience} epochs)"
                )
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, best_val_loss


def evaluate_predictions(trues, preds):
    """Return per-step and overall metrics"""
    step_metrics = []
    for step in range(PRED_LEN):
        true_step = trues[:, step]
        pred_step = preds[:, step]

        mse_step = mean_squared_error(true_step, pred_step)
        mae_step = mean_absolute_error(true_step, pred_step)
        rmse_step = math.sqrt(mse_step)
        r2_step = r2_score(true_step, pred_step)

        step_metrics.append({
            "step": step + 1,
            "mse": mse_step,
            "mae": mae_step,
            "rmse": rmse_step,
            "r2": r2_step
        })

    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(trues, preds)

    overall = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    return step_metrics, overall


def generate_and_save_predictions(
    model,
    test_loader,
    scaler,
    dates_work,
    train_size,
    look_back,
    save_path,
    combo_name=""
):
    model.eval()
    preds_scaled, trues_scaled = [], []

    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx = bx.to(device)
            pred = model(bx).cpu().numpy()
            preds_scaled.append(pred)
            trues_scaled.append(by.numpy())

    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues_scaled = np.concatenate(trues_scaled, axis=0)

    preds = denormalize_predictions(preds_scaled, scaler)
    trues = denormalize_predictions(trues_scaled, scaler)

    res_len = preds.shape[0]
    start_idx = train_size + 2 * look_back
    res_dates = dates_work[start_idx: start_idx + res_len]

    step_metrics, overall = evaluate_predictions(trues, preds)

    print("\n" + "=" * 80)
    print(f"{combo_name}  (Step 1 ~ Step 12)")
    print("=" * 80)
    for item in step_metrics:
        print(
            f"Step {item['step']} - "
            f"MSE: {item['mse']:.6f}, "
            f"MAE: {item['mae']:.6f}, "
            f"RMSE: {item['rmse']:.6f}, "
            f"R2: {item['r2']:.6f}"
        )

    print("\n" + "=" * 80)
    print(f"{combo_name} Overall")
    print("=" * 80)
    print(
        f"Advanced Model - "
        f"MSE: {overall['mse']:.6f}, "
        f"MAE: {overall['mae']:.6f}, "
        f"RMSE: {overall['rmse']:.6f}, "
        f"R2: {overall['r2']:.6f}"
    )

    # save prediction csv
    df_out = pd.DataFrame()
    df_out["datetime"] = res_dates
    for k in range(preds.shape[1]):
        df_out[f"pred_step_{k + 1}"] = preds[:, k]
    df_out["pred_avg_1hr"] = preds.mean(axis=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding="utf-8")
    print(f"\nsave to: {save_path}")

    return {
        "step_metrics": step_metrics,
        "overall_metrics": overall,
        "preds": preds,
        "trues": trues
    }


def run_single_experiment(
    df,
    data_file,
    output_dir,
    model_dir,
    pretrained_path,
    look_back_override,
    auto_coef,
    nhits_coef
):
    print("\n" + "#" * 100)
    print(f"Current combination -> auto_coef={auto_coef}, nhits_coef={nhits_coef}")
    print("#" * 100)

    lb_def = 48 if "30" in os.path.basename(data_file) else 20
    look_back = look_back_override if look_back_override is not None else lb_def
    print(f"length (look_back): {look_back}")

    train_loader, test_loader, scaler, dates_work, train_size, look_back = preprocess_and_split_data(
        df, look_back, PRED_LEN
    )
    print(
        f"Data preprocessing completed - "
        f"Training set batches: {len(train_loader)}, "
        f"Test set batches: {len(test_loader)}"
    )

    model = DENet(
        seq_len=look_back,
        pred_len=PRED_LEN,
        features=6,
        dropout=0.3,
        nhits_hidden_ratio=4,
        nhits_coef=nhits_coef,
        auto_coef=auto_coef
    ).to(device)

    print(f"finished: {sum(p.numel() for p in model.parameters()):,}")

    if pretrained_path is not None and os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
            print(f"Successfully loaded pretrained weights: {pretrained_path}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")

    combo_tag = f"auto_{auto_coef}_nhits_{nhits_coef}".replace(".", "p")
    model_name = f"DENet_{combo_tag}.pth"

    model, best_val_loss = train_model(
        model,
        train_loader,
        test_loader,
        epochs=180,
        lr=5e-4,
        model_dir=model_dir,
        model_name=model_name
    )

    save_path = os.path.join(output_dir, f"pred_{combo_tag}.csv")
    result = generate_and_save_predictions(
        model=model,
        test_loader=test_loader,
        scaler=scaler,
        dates_work=dates_work,
        train_size=train_size,
        look_back=look_back,
        save_path=save_path,
        combo_name=f"[auto={auto_coef}, nhits={nhits_coef}]"
    )

    summary = {
        "auto_coef": auto_coef,
        "nhits_coef": nhits_coef,
        "best_val_loss": best_val_loss,
        "mse": result["overall_metrics"]["mse"],
        "mae": result["overall_metrics"]["mae"],
        "rmse": result["overall_metrics"]["rmse"],
        "r2": result["overall_metrics"]["r2"],
        "pred_path": save_path,
        "model_path": os.path.join(model_dir, model_name)
    }

    for step_item in result["step_metrics"]:
        s = step_item["step"]
        summary[f"step{s}_mse"] = step_item["mse"]
        summary[f"step{s}_mae"] = step_item["mae"]
        summary[f"step{s}_rmse"] = step_item["rmse"]
        summary[f"step{s}_r2"] = step_item["r2"]

    return summary


# =============================== Main ===============================
def main(
    data_file="JM_5.csv",
    output_dir="results_DENet",
    model_dir="models_DENet",
    pretrained_path: str | None = None,
    look_back_override: int | None = None,
    auto_grid=None,
    nhits_grid=None
):
    """
    Main function:
    Data loading -> Preprocessing -> Grid Search (fixed fusion weights) -> Prediction -> Result saving

    Parameters:
    - data_file: Path to input csv
    - output_dir: Directory for prediction results
    - model_dir: Directory for model weights
    - pretrained_path: Optional pretrained model path
    - look_back_override: Optional look_back
    - auto_grid: grid candidates for auto branch fusion weight
    - nhits_grid: grid candidates for nhits branch fusion weight
    """
    print(f"device: {device}")
    print(f"file: {data_file}")

    try:
        df = pd.read_csv(data_file)
        print(f"Data loaded successfully, total rows: {len(df)}")
    except FileNotFoundError:
        print(f"fail '{data_file}'")
        return
    except Exception as e:
        print(f"fail: {e}")
        return

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].isnull().any():
        print("warning: invalid datetime found, dropping invalid rows")
        df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    # default grids
    if auto_grid is None:
        auto_grid = [0.0, 0.05, 0.1, 0.2, 0.5]
    if nhits_grid is None:
        nhits_grid = [0.0, 0.001, 0.01, 0.05, 0.1]

    print(f"auto_grid: {auto_grid}")
    print(f"nhits_grid: {nhits_grid}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    all_results = []
    best_result = None
    best_rmse = float("inf")

    grid_combinations = list(itertools.product(auto_grid, nhits_grid))
    print(f"Total grid combinations: {len(grid_combinations)}")

    for idx, (auto_coef, nhits_coef) in enumerate(grid_combinations, start=1):
        print("\n" + "=" * 120)
        print(f"Grid Search Progress: {idx}/{len(grid_combinations)}")
        print("=" * 120)

        try:
            result = run_single_experiment(
                df=df,
                data_file=data_file,
                output_dir=output_dir,
                model_dir=model_dir,
                pretrained_path=pretrained_path,
                look_back_override=look_back_override,
                auto_coef=auto_coef,
                nhits_coef=nhits_coef
            )
            all_results.append(result)

            print("\nCurrent combination summary:")
            print(
                f"auto_coef={auto_coef}, nhits_coef={nhits_coef}, "
                f"best_val_loss={result['best_val_loss']:.6f}, "
                f"MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, "
                f"RMSE={result['rmse']:.6f}, R2={result['r2']:.6f}"
            )

            if result["rmse"] < best_rmse:
                best_rmse = result["rmse"]
                best_result = result

        except Exception as e:
            print(f"Combination failed: auto_coef={auto_coef}, nhits_coef={nhits_coef}, error={e}")

    if len(all_results) == 0:
        print("No successful grid-search result.")
        return

    # save grid summary
    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values(by=["rmse", "mae", "mse"], ascending=[True, True, True]).reset_index(drop=True)

    grid_result_path = os.path.join(output_dir, "grid_search_results.csv")
    result_df.to_csv(grid_result_path, index=False, encoding="utf-8")
    print(f"\nGrid search summary saved to: {grid_result_path}")

    print("\n" + "#" * 120)
    print("Grid Search Final Ranking (Top 10)")
    print("#" * 120)
    print(
        result_df[
            ["auto_coef", "nhits_coef", "best_val_loss", "mse", "mae", "rmse", "r2"]
        ].head(10).to_string(index=False)
    )

    if best_result is not None:
        print("\n" + "#" * 120)
        print("Best Combination")
        print("#" * 120)
        print(
            f"auto_coef={best_result['auto_coef']}, "
            f"nhits_coef={best_result['nhits_coef']}, "
            f"best_val_loss={best_result['best_val_loss']:.6f}, "
            f"MSE={best_result['mse']:.6f}, "
            f"MAE={best_result['mae']:.6f}, "
            f"RMSE={best_result['rmse']:.6f}, "
            f"R2={best_result['r2']:.6f}"
        )

        # copy best model path info
        best_info_path = os.path.join(output_dir, "best_combination.txt")
        with open(best_info_path, "w", encoding="utf-8") as f:
            f.write("Best Combination\n")
            f.write(f"auto_coef={best_result['auto_coef']}\n")
            f.write(f"nhits_coef={best_result['nhits_coef']}\n")
            f.write(f"best_val_loss={best_result['best_val_loss']:.6f}\n")
            f.write(f"MSE={best_result['mse']:.6f}\n")
            f.write(f"MAE={best_result['mae']:.6f}\n")
            f.write(f"RMSE={best_result['rmse']:.6f}\n")
            f.write(f"R2={best_result['r2']:.6f}\n")
            f.write(f"pred_path={best_result['pred_path']}\n")
            f.write(f"model_path={best_result['model_path']}\n")
        print(f"Best combination info saved to: {best_info_path}")

    print("\nfinished")
    return None


# =============================== Enter ===============================
if __name__ == "__main__":
    import sys
    import ast

    # Usage:
    # python script.py [data_file] [output_dir] [model_dir] [pretrained_path] [look_back] [auto_grid] [nhits_grid]
    #
    # Example:
    # python script.py JM_5.csv results_DENet models_DENet None 20 "[0,0.05,0.1,0.2,0.5]" "[0,0.001,0.01,0.05,0.1]"

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "results_DENet"
        model_dir = sys.argv[3] if len(sys.argv) > 3 else "models_DENet"
        pretrained = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "None" else None
        look_back_arg = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in ("", "None") else None

        auto_grid_arg = None
        nhits_grid_arg = None

        if len(sys.argv) > 6 and sys.argv[6] not in ("", "None"):
            auto_grid_arg = ast.literal_eval(sys.argv[6])

        if len(sys.argv) > 7 and sys.argv[7] not in ("", "None"):
            nhits_grid_arg = ast.literal_eval(sys.argv[7])

        main(
            data_file=data_file,
            output_dir=output_dir,
            model_dir=model_dir,
            pretrained_path=pretrained,
            look_back_override=look_back_arg,
            auto_grid=auto_grid_arg,
            nhits_grid=nhits_grid_arg
        )
    else:
        main()
"""

"""