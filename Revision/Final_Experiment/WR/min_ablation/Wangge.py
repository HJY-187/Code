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
    epochs=160,
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
    data_file="WR_5.csv",
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
    # python script.py WR_5.csv results_DENet models_DENet None 20 "[0,0.05,0.1,0.2,0.5]" "[0,0.001,0.01,0.05,0.1]"

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
D:\miniconda\envs\Py123\python.exe D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py 
device: cuda
file: WR_5.csv
Data loaded successfully, total rows: 19530
auto_grid: [0.0, 0.05, 0.1, 0.2, 0.5]
nhits_grid: [0.0, 0.001, 0.01, 0.05, 0.1]
Total grid combinations: 25

========================================================================================================================
Grid Search Progress: 1/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.0, nhits_coef=0.0
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.121916 | Val Loss: 0.014590
Epoch 21/180 - Train Loss: 0.002856 | Val Loss: 0.007178
Epoch 41/180 - Train Loss: 0.002535 | Val Loss: 0.006176
Epoch 61/180 - Train Loss: 0.002449 | Val Loss: 0.005884
Epoch 81/180 - Train Loss: 0.002418 | Val Loss: 0.005850
Early stop at epoch 88 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.0, nhits=0.0]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 280.600891, MAE: 6.797659, RMSE: 16.751146, R2: 0.994285
Step 2 - MSE: 474.704712, MAE: 8.576560, RMSE: 21.787719, R2: 0.990326
Step 3 - MSE: 661.828064, MAE: 10.470488, RMSE: 25.726019, R2: 0.986501
Step 4 - MSE: 829.906860, MAE: 12.081967, RMSE: 28.808104, R2: 0.983060
Step 5 - MSE: 1016.394226, MAE: 13.741944, RMSE: 31.880938, R2: 0.979236
Step 6 - MSE: 1193.853638, MAE: 15.047048, RMSE: 34.552187, R2: 0.975592
Step 7 - MSE: 1355.135498, MAE: 16.434196, RMSE: 36.812165, R2: 0.972273
Step 8 - MSE: 1512.284058, MAE: 17.696726, RMSE: 38.888097, R2: 0.969032
Step 9 - MSE: 1683.624878, MAE: 18.554581, RMSE: 41.031998, R2: 0.965494
Step 10 - MSE: 1842.675049, MAE: 19.787930, RMSE: 42.926391, R2: 0.962201
Step 11 - MSE: 2004.493896, MAE: 20.811979, RMSE: 44.771575, R2: 0.958846
Step 12 - MSE: 2175.707764, MAE: 22.113722, RMSE: 46.644483, R2: 0.955293

================================================================================
[auto=0.0, nhits=0.0] Overall
================================================================================
Advanced Model - MSE: 1252.600464, MAE: 15.176247, RMSE: 35.392096, R2: 0.974345

save to: results_DENet\pred_auto_0p0_nhits_0p0.csv

Current combination summary:
auto_coef=0.0, nhits_coef=0.0, best_val_loss=0.005770, MSE=1252.600464, MAE=15.176247, RMSE=35.392096, R2=0.974345

========================================================================================================================
Grid Search Progress: 2/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.0, nhits_coef=0.001
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.107280 | Val Loss: 0.012247
Epoch 21/180 - Train Loss: 0.002736 | Val Loss: 0.006692
Epoch 41/180 - Train Loss: 0.002504 | Val Loss: 0.006029
Epoch 61/180 - Train Loss: 0.002449 | Val Loss: 0.005874
Epoch 81/180 - Train Loss: 0.002434 | Val Loss: 0.005868
Epoch 101/180 - Train Loss: 0.002374 | Val Loss: 0.005720
Epoch 121/180 - Train Loss: 0.002368 | Val Loss: 0.005705
Epoch 141/180 - Train Loss: 0.002377 | Val Loss: 0.005692
Early stop at epoch 153 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.0, nhits=0.001]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 273.265625, MAE: 5.963037, RMSE: 16.530748, R2: 0.994435
Step 2 - MSE: 464.608307, MAE: 8.388554, RMSE: 21.554775, R2: 0.990531
Step 3 - MSE: 646.752808, MAE: 10.299866, RMSE: 25.431335, R2: 0.986808
Step 4 - MSE: 827.950134, MAE: 11.957016, RMSE: 28.774123, R2: 0.983100
Step 5 - MSE: 1013.429565, MAE: 13.505630, RMSE: 31.834409, R2: 0.979296
Step 6 - MSE: 1191.169312, MAE: 14.892062, RMSE: 34.513321, R2: 0.975647
Step 7 - MSE: 1354.594482, MAE: 16.205782, RMSE: 36.804816, R2: 0.972284
Step 8 - MSE: 1506.095215, MAE: 17.423630, RMSE: 38.808443, R2: 0.969159
Step 9 - MSE: 1676.566528, MAE: 18.495626, RMSE: 40.945898, R2: 0.965639
Step 10 - MSE: 1835.803223, MAE: 19.781158, RMSE: 42.846274, R2: 0.962342
Step 11 - MSE: 1994.466431, MAE: 20.746605, RMSE: 44.659450, R2: 0.959052
Step 12 - MSE: 2160.402588, MAE: 21.813204, RMSE: 46.480131, R2: 0.955608

================================================================================
[auto=0.0, nhits=0.001] Overall
================================================================================
Advanced Model - MSE: 1245.424683, MAE: 14.956008, RMSE: 35.290575, R2: 0.974492

save to: results_DENet\pred_auto_0p0_nhits_0p001.csv

Current combination summary:
auto_coef=0.0, nhits_coef=0.001, best_val_loss=0.005692, MSE=1245.424683, MAE=14.956008, RMSE=35.290575, R2=0.974492

========================================================================================================================
Grid Search Progress: 3/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.0, nhits_coef=0.01
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.053436 | Val Loss: 0.012427
Epoch 21/180 - Train Loss: 0.002758 | Val Loss: 0.006792
Epoch 41/180 - Train Loss: 0.002522 | Val Loss: 0.006068
Epoch 61/180 - Train Loss: 0.002455 | Val Loss: 0.005860
Epoch 81/180 - Train Loss: 0.002394 | Val Loss: 0.005793
Epoch 101/180 - Train Loss: 0.002388 | Val Loss: 0.005898
Epoch 121/180 - Train Loss: 0.002382 | Val Loss: 0.005786
Epoch 141/180 - Train Loss: 0.002374 | Val Loss: 0.005789
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))
Early stop at epoch 155 (val loss no improvement for 12 epochs)

================================================================================
[auto=0.0, nhits=0.01]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 275.969727, MAE: 6.162525, RMSE: 16.612337, R2: 0.994380
Step 2 - MSE: 470.517944, MAE: 8.329204, RMSE: 21.691426, R2: 0.990411
Step 3 - MSE: 651.917725, MAE: 10.220628, RMSE: 25.532680, R2: 0.986703
Step 4 - MSE: 836.002441, MAE: 11.879539, RMSE: 28.913707, R2: 0.982935
Step 5 - MSE: 1016.018982, MAE: 13.453951, RMSE: 31.875053, R2: 0.979243
Step 6 - MSE: 1192.501953, MAE: 14.968234, RMSE: 34.532622, R2: 0.975619
Step 7 - MSE: 1357.156616, MAE: 16.228281, RMSE: 36.839607, R2: 0.972231
Step 8 - MSE: 1508.300537, MAE: 17.353710, RMSE: 38.836845, R2: 0.969114
Step 9 - MSE: 1675.627808, MAE: 18.611748, RMSE: 40.934433, R2: 0.965658
Step 10 - MSE: 1835.951172, MAE: 19.719776, RMSE: 42.848001, R2: 0.962339
Step 11 - MSE: 1995.300293, MAE: 20.787743, RMSE: 44.668784, R2: 0.959035
Step 12 - MSE: 2170.642822, MAE: 21.887655, RMSE: 46.590158, R2: 0.955397

================================================================================
[auto=0.0, nhits=0.01] Overall
================================================================================
Advanced Model - MSE: 1248.825317, MAE: 14.966914, RMSE: 35.338723, R2: 0.974422

save to: results_DENet\pred_auto_0p0_nhits_0p01.csv

Current combination summary:
auto_coef=0.0, nhits_coef=0.01, best_val_loss=0.005717, MSE=1248.825317, MAE=14.966914, RMSE=35.338723, R2=0.974422

========================================================================================================================
Grid Search Progress: 4/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.0, nhits_coef=0.05
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.093970 | Val Loss: 0.011941
Epoch 21/180 - Train Loss: 0.002786 | Val Loss: 0.006832
Epoch 41/180 - Train Loss: 0.002524 | Val Loss: 0.006299
Epoch 61/180 - Train Loss: 0.002424 | Val Loss: 0.005840
Epoch 81/180 - Train Loss: 0.002406 | Val Loss: 0.005786
Epoch 101/180 - Train Loss: 0.002386 | Val Loss: 0.005795
Epoch 121/180 - Train Loss: 0.002380 | Val Loss: 0.005758
Epoch 141/180 - Train Loss: 0.002365 | Val Loss: 0.005760
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))
Early stop at epoch 142 (val loss no improvement for 12 epochs)

================================================================================
[auto=0.0, nhits=0.05]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 272.588776, MAE: 5.988661, RMSE: 16.510263, R2: 0.994449
Step 2 - MSE: 466.488617, MAE: 8.273119, RMSE: 21.598348, R2: 0.990493
Step 3 - MSE: 655.375916, MAE: 10.220936, RMSE: 25.600311, R2: 0.986633
Step 4 - MSE: 832.151978, MAE: 11.880380, RMSE: 28.847045, R2: 0.983014
Step 5 - MSE: 1018.299500, MAE: 13.672989, RMSE: 31.910805, R2: 0.979197
Step 6 - MSE: 1189.817627, MAE: 14.981351, RMSE: 34.493733, R2: 0.975674
Step 7 - MSE: 1354.777588, MAE: 16.303720, RMSE: 36.807303, R2: 0.972280
Step 8 - MSE: 1508.863525, MAE: 17.484871, RMSE: 38.844093, R2: 0.969102
Step 9 - MSE: 1675.550659, MAE: 18.674887, RMSE: 40.933491, R2: 0.965659
Step 10 - MSE: 1841.797485, MAE: 19.851048, RMSE: 42.916168, R2: 0.962219
Step 11 - MSE: 1994.773804, MAE: 20.757767, RMSE: 44.662891, R2: 0.959046
Step 12 - MSE: 2159.400391, MAE: 21.917034, RMSE: 46.469349, R2: 0.955628

================================================================================
[auto=0.0, nhits=0.05] Overall
================================================================================
Advanced Model - MSE: 1247.489258, MAE: 15.000566, RMSE: 35.319814, R2: 0.974449

save to: results_DENet\pred_auto_0p0_nhits_0p05.csv

Current combination summary:
auto_coef=0.0, nhits_coef=0.05, best_val_loss=0.005720, MSE=1247.489258, MAE=15.000566, RMSE=35.319814, R2=0.974449

========================================================================================================================
Grid Search Progress: 5/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.0, nhits_coef=0.1
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.099351 | Val Loss: 0.014483
Epoch 21/180 - Train Loss: 0.002805 | Val Loss: 0.007081
Epoch 41/180 - Train Loss: 0.002530 | Val Loss: 0.006213
Epoch 61/180 - Train Loss: 0.002467 | Val Loss: 0.006190
Epoch 81/180 - Train Loss: 0.002430 | Val Loss: 0.005816
Epoch 101/180 - Train Loss: 0.002381 | Val Loss: 0.005757
Epoch 121/180 - Train Loss: 0.002385 | Val Loss: 0.005771
Epoch 141/180 - Train Loss: 0.002369 | Val Loss: 0.005722
Epoch 161/180 - Train Loss: 0.002363 | Val Loss: 0.005716
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))
Early stop at epoch 175 (val loss no improvement for 12 epochs)

================================================================================
[auto=0.0, nhits=0.1]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 275.754913, MAE: 6.126553, RMSE: 16.605870, R2: 0.994384
Step 2 - MSE: 468.742920, MAE: 8.307577, RMSE: 21.650472, R2: 0.990447
Step 3 - MSE: 652.958374, MAE: 10.226798, RMSE: 25.553050, R2: 0.986682
Step 4 - MSE: 832.743774, MAE: 11.903735, RMSE: 28.857300, R2: 0.983002
Step 5 - MSE: 1017.015381, MAE: 13.534669, RMSE: 31.890679, R2: 0.979223
Step 6 - MSE: 1193.001831, MAE: 14.939945, RMSE: 34.539859, R2: 0.975609
Step 7 - MSE: 1354.915894, MAE: 16.206303, RMSE: 36.809182, R2: 0.972277
Step 8 - MSE: 1508.096802, MAE: 17.386005, RMSE: 38.834222, R2: 0.969118
Step 9 - MSE: 1675.138794, MAE: 18.590076, RMSE: 40.928459, R2: 0.965668
Step 10 - MSE: 1836.230225, MAE: 19.757710, RMSE: 42.851257, R2: 0.962333
Step 11 - MSE: 1995.228149, MAE: 20.772894, RMSE: 44.667977, R2: 0.959037
Step 12 - MSE: 2162.697754, MAE: 21.859404, RMSE: 46.504814, R2: 0.955560

================================================================================
[auto=0.0, nhits=0.1] Overall
================================================================================
Advanced Model - MSE: 1247.709473, MAE: 14.967637, RMSE: 35.322931, R2: 0.974445

save to: results_DENet\pred_auto_0p0_nhits_0p1.csv

Current combination summary:
auto_coef=0.0, nhits_coef=0.1, best_val_loss=0.005707, MSE=1247.709473, MAE=14.967637, RMSE=35.322931, R2=0.974445

========================================================================================================================
Grid Search Progress: 6/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.05, nhits_coef=0.0
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.100631 | Val Loss: 0.013180
Epoch 21/180 - Train Loss: 0.002681 | Val Loss: 0.006566
Epoch 41/180 - Train Loss: 0.002485 | Val Loss: 0.006139
Epoch 61/180 - Train Loss: 0.002436 | Val Loss: 0.006106
Epoch 81/180 - Train Loss: 0.002428 | Val Loss: 0.005942
Epoch 101/180 - Train Loss: 0.002369 | Val Loss: 0.005873
Epoch 121/180 - Train Loss: 0.002352 | Val Loss: 0.005761
Early stop at epoch 125 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.05, nhits=0.0]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 279.478058, MAE: 6.143159, RMSE: 16.717597, R2: 0.994308
Step 2 - MSE: 472.892944, MAE: 8.344547, RMSE: 21.746102, R2: 0.990362
Step 3 - MSE: 653.687805, MAE: 10.288250, RMSE: 25.567319, R2: 0.986667
Step 4 - MSE: 839.511536, MAE: 11.934803, RMSE: 28.974325, R2: 0.982864
Step 5 - MSE: 1020.102783, MAE: 13.510290, RMSE: 31.939048, R2: 0.979160
Step 6 - MSE: 1196.911255, MAE: 14.960139, RMSE: 34.596405, R2: 0.975529
Step 7 - MSE: 1360.598267, MAE: 16.243145, RMSE: 36.886288, R2: 0.972161
Step 8 - MSE: 1517.156982, MAE: 17.367304, RMSE: 38.950699, R2: 0.968933
Step 9 - MSE: 1678.227173, MAE: 18.556252, RMSE: 40.966171, R2: 0.965605
Step 10 - MSE: 1839.812988, MAE: 19.636692, RMSE: 42.893041, R2: 0.962260
Step 11 - MSE: 2006.955322, MAE: 20.759420, RMSE: 44.799055, R2: 0.958796
Step 12 - MSE: 2176.468750, MAE: 21.866219, RMSE: 46.652639, R2: 0.955277

================================================================================
[auto=0.05, nhits=0.0] Overall
================================================================================
Advanced Model - MSE: 1253.482056, MAE: 14.967519, RMSE: 35.404549, R2: 0.974327

save to: results_DENet\pred_auto_0p05_nhits_0p0.csv

Current combination summary:
auto_coef=0.05, nhits_coef=0.0, best_val_loss=0.005748, MSE=1253.482056, MAE=14.967519, RMSE=35.404549, R2=0.974327

========================================================================================================================
Grid Search Progress: 7/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.05, nhits_coef=0.001
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.121752 | Val Loss: 0.020908
Epoch 21/180 - Train Loss: 0.002755 | Val Loss: 0.006650
Epoch 41/180 - Train Loss: 0.002502 | Val Loss: 0.006074
Epoch 61/180 - Train Loss: 0.002445 | Val Loss: 0.006001
Early stop at epoch 79 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.05, nhits=0.001]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 288.675171, MAE: 6.297095, RMSE: 16.990444, R2: 0.994121
Step 2 - MSE: 488.789825, MAE: 8.583225, RMSE: 22.108592, R2: 0.990038
Step 3 - MSE: 673.627808, MAE: 10.501755, RMSE: 25.954341, R2: 0.986260
Step 4 - MSE: 860.333252, MAE: 12.358431, RMSE: 29.331438, R2: 0.982439
Step 5 - MSE: 1040.868408, MAE: 13.803211, RMSE: 32.262492, R2: 0.978736
Step 6 - MSE: 1208.582153, MAE: 15.378263, RMSE: 34.764668, R2: 0.975291
Step 7 - MSE: 1373.874268, MAE: 16.663244, RMSE: 37.065810, R2: 0.971889
Step 8 - MSE: 1541.666260, MAE: 17.793331, RMSE: 39.264058, R2: 0.968431
Step 9 - MSE: 1699.114136, MAE: 18.722832, RMSE: 41.220312, R2: 0.965177
Step 10 - MSE: 1852.300537, MAE: 19.982939, RMSE: 43.038361, R2: 0.962004
Step 11 - MSE: 2018.118408, MAE: 20.921761, RMSE: 44.923473, R2: 0.958567
Step 12 - MSE: 2192.842041, MAE: 22.047041, RMSE: 46.827791, R2: 0.954941

================================================================================
[auto=0.05, nhits=0.001] Overall
================================================================================
Advanced Model - MSE: 1269.898071, MAE: 15.254425, RMSE: 35.635629, R2: 0.973991

save to: results_DENet\pred_auto_0p05_nhits_0p001.csv

Current combination summary:
auto_coef=0.05, nhits_coef=0.001, best_val_loss=0.005893, MSE=1269.898071, MAE=15.254425, RMSE=35.635629, R2=0.973991

========================================================================================================================
Grid Search Progress: 8/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.05, nhits_coef=0.01
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.105615 | Val Loss: 0.014151
Epoch 21/180 - Train Loss: 0.002780 | Val Loss: 0.006842
Epoch 41/180 - Train Loss: 0.002494 | Val Loss: 0.006112
Epoch 61/180 - Train Loss: 0.002453 | Val Loss: 0.006135
Epoch 81/180 - Train Loss: 0.002410 | Val Loss: 0.005967
Epoch 101/180 - Train Loss: 0.002377 | Val Loss: 0.005870
Epoch 121/180 - Train Loss: 0.002361 | Val Loss: 0.005760
Early stop at epoch 135 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.05, nhits=0.01]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 277.111420, MAE: 6.112954, RMSE: 16.646664, R2: 0.994357
Step 2 - MSE: 474.054443, MAE: 8.931353, RMSE: 21.772791, R2: 0.990339
Step 3 - MSE: 653.477112, MAE: 10.597538, RMSE: 25.563198, R2: 0.986671
Step 4 - MSE: 836.468933, MAE: 12.406138, RMSE: 28.921773, R2: 0.982926
Step 5 - MSE: 1014.964355, MAE: 13.651362, RMSE: 31.858505, R2: 0.979265
Step 6 - MSE: 1190.915161, MAE: 14.971947, RMSE: 34.509639, R2: 0.975652
Step 7 - MSE: 1353.217529, MAE: 16.394041, RMSE: 36.786105, R2: 0.972312
Step 8 - MSE: 1512.660645, MAE: 17.353624, RMSE: 38.892938, R2: 0.969025
Step 9 - MSE: 1679.354248, MAE: 18.805717, RMSE: 40.979925, R2: 0.965582
Step 10 - MSE: 1835.487061, MAE: 19.740046, RMSE: 42.842585, R2: 0.962349
Step 11 - MSE: 2004.254883, MAE: 20.967833, RMSE: 44.768905, R2: 0.958851
Step 12 - MSE: 2169.717773, MAE: 21.927488, RMSE: 46.580229, R2: 0.955416

================================================================================
[auto=0.05, nhits=0.01] Overall
================================================================================
Advanced Model - MSE: 1250.139893, MAE: 15.155011, RMSE: 35.357317, R2: 0.974395

save to: results_DENet\pred_auto_0p05_nhits_0p01.csv

Current combination summary:
auto_coef=0.05, nhits_coef=0.01, best_val_loss=0.005750, MSE=1250.139893, MAE=15.155011, RMSE=35.357317, R2=0.974395

========================================================================================================================
Grid Search Progress: 9/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.05, nhits_coef=0.05
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.097779 | Val Loss: 0.013558
Epoch 21/180 - Train Loss: 0.002691 | Val Loss: 0.006894
Epoch 41/180 - Train Loss: 0.002481 | Val Loss: 0.006261
Epoch 61/180 - Train Loss: 0.002416 | Val Loss: 0.005971
Epoch 81/180 - Train Loss: 0.002416 | Val Loss: 0.005962
Epoch 101/180 - Train Loss: 0.002373 | Val Loss: 0.005802
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))
Early stop at epoch 104 (val loss no improvement for 12 epochs)

================================================================================
[auto=0.05, nhits=0.05]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 281.202301, MAE: 6.508224, RMSE: 16.769088, R2: 0.994273
Step 2 - MSE: 475.143677, MAE: 8.811365, RMSE: 21.797791, R2: 0.990317
Step 3 - MSE: 658.210327, MAE: 10.243173, RMSE: 25.655610, R2: 0.986575
Step 4 - MSE: 839.748535, MAE: 12.218101, RMSE: 28.978415, R2: 0.982859
Step 5 - MSE: 1017.550476, MAE: 13.894386, RMSE: 31.899067, R2: 0.979212
Step 6 - MSE: 1197.736450, MAE: 15.182043, RMSE: 34.608329, R2: 0.975512
Step 7 - MSE: 1357.946045, MAE: 16.367424, RMSE: 36.850319, R2: 0.972215
Step 8 - MSE: 1512.541626, MAE: 17.717243, RMSE: 38.891408, R2: 0.969027
Step 9 - MSE: 1684.735840, MAE: 18.768154, RMSE: 41.045534, R2: 0.965471
Step 10 - MSE: 1838.752686, MAE: 19.793608, RMSE: 42.880680, R2: 0.962282
Step 11 - MSE: 2006.905029, MAE: 20.763306, RMSE: 44.798494, R2: 0.958797
Step 12 - MSE: 2182.780518, MAE: 22.035179, RMSE: 46.720237, R2: 0.955148

================================================================================
[auto=0.05, nhits=0.05] Overall
================================================================================
Advanced Model - MSE: 1254.437134, MAE: 15.191853, RMSE: 35.418034, R2: 0.974307

save to: results_DENet\pred_auto_0p05_nhits_0p05.csv

Current combination summary:
auto_coef=0.05, nhits_coef=0.05, best_val_loss=0.005785, MSE=1254.437134, MAE=15.191853, RMSE=35.418034, R2=0.974307

========================================================================================================================
Grid Search Progress: 10/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.05, nhits_coef=0.1
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.103318 | Val Loss: 0.016570
Epoch 21/180 - Train Loss: 0.002692 | Val Loss: 0.006891
Epoch 41/180 - Train Loss: 0.002510 | Val Loss: 0.006163
Epoch 61/180 - Train Loss: 0.002439 | Val Loss: 0.005901
Epoch 81/180 - Train Loss: 0.002408 | Val Loss: 0.005832
Epoch 101/180 - Train Loss: 0.002388 | Val Loss: 0.005807
Epoch 121/180 - Train Loss: 0.002376 | Val Loss: 0.005765
Epoch 141/180 - Train Loss: 0.002361 | Val Loss: 0.005979
Epoch 161/180 - Train Loss: 0.002349 | Val Loss: 0.005798
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.05, nhits=0.1]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 274.998962, MAE: 6.061992, RMSE: 16.583093, R2: 0.994400
Step 2 - MSE: 467.893585, MAE: 8.385304, RMSE: 21.630848, R2: 0.990464
Step 3 - MSE: 650.809875, MAE: 10.167749, RMSE: 25.510976, R2: 0.986726
Step 4 - MSE: 832.267944, MAE: 11.878554, RMSE: 28.849054, R2: 0.983011
Step 5 - MSE: 1013.428345, MAE: 13.529756, RMSE: 31.834389, R2: 0.979296
Step 6 - MSE: 1191.778320, MAE: 14.942814, RMSE: 34.522142, R2: 0.975634
Step 7 - MSE: 1355.590820, MAE: 16.227985, RMSE: 36.818349, R2: 0.972263
Step 8 - MSE: 1510.971924, MAE: 17.391775, RMSE: 38.871222, R2: 0.969059
Step 9 - MSE: 1677.157593, MAE: 18.603100, RMSE: 40.953115, R2: 0.965627
Step 10 - MSE: 1831.352661, MAE: 19.770180, RMSE: 42.794306, R2: 0.962433
Step 11 - MSE: 1999.354248, MAE: 20.712755, RMSE: 44.714139, R2: 0.958952
Step 12 - MSE: 2165.803223, MAE: 21.852890, RMSE: 46.538191, R2: 0.955497

================================================================================
[auto=0.05, nhits=0.1] Overall
================================================================================
Advanced Model - MSE: 1247.615845, MAE: 14.960414, RMSE: 35.321606, R2: 0.974447

save to: results_DENet\pred_auto_0p05_nhits_0p1.csv

Current combination summary:
auto_coef=0.05, nhits_coef=0.1, best_val_loss=0.005711, MSE=1247.615845, MAE=14.960414, RMSE=35.321606, R2=0.974447

========================================================================================================================
Grid Search Progress: 11/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.1, nhits_coef=0.0
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.083230 | Val Loss: 0.012415
Epoch 21/180 - Train Loss: 0.002664 | Val Loss: 0.006576
Epoch 41/180 - Train Loss: 0.002485 | Val Loss: 0.006979
Epoch 61/180 - Train Loss: 0.002422 | Val Loss: 0.005866
Epoch 81/180 - Train Loss: 0.002395 | Val Loss: 0.006271
Epoch 101/180 - Train Loss: 0.002370 | Val Loss: 0.005797
Epoch 121/180 - Train Loss: 0.002354 | Val Loss: 0.005770
Early stop at epoch 123 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.1, nhits=0.0]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 279.428894, MAE: 6.316441, RMSE: 16.716127, R2: 0.994309
Step 2 - MSE: 469.239349, MAE: 8.457615, RMSE: 21.661933, R2: 0.990437
Step 3 - MSE: 654.770020, MAE: 10.303285, RMSE: 25.588474, R2: 0.986645
Step 4 - MSE: 838.515381, MAE: 11.945838, RMSE: 28.957130, R2: 0.982884
Step 5 - MSE: 1016.472534, MAE: 13.497123, RMSE: 31.882166, R2: 0.979234
Step 6 - MSE: 1192.624634, MAE: 14.909496, RMSE: 34.534398, R2: 0.975617
Step 7 - MSE: 1359.964111, MAE: 16.522778, RMSE: 36.877691, R2: 0.972174
Step 8 - MSE: 1508.461060, MAE: 17.500082, RMSE: 38.838912, R2: 0.969111
Step 9 - MSE: 1681.568359, MAE: 18.586922, RMSE: 41.006931, R2: 0.965536
Step 10 - MSE: 1845.721069, MAE: 19.792543, RMSE: 42.961856, R2: 0.962139
Step 11 - MSE: 2010.212646, MAE: 20.826527, RMSE: 44.835395, R2: 0.958729
Step 12 - MSE: 2177.221436, MAE: 21.957962, RMSE: 46.660705, R2: 0.955262

================================================================================
[auto=0.1, nhits=0.0] Overall
================================================================================
Advanced Model - MSE: 1252.848267, MAE: 15.051381, RMSE: 35.395597, R2: 0.974340

save to: results_DENet\pred_auto_0p1_nhits_0p0.csv

Current combination summary:
auto_coef=0.1, nhits_coef=0.0, best_val_loss=0.005763, MSE=1252.848267, MAE=15.051381, RMSE=35.395597, R2=0.974340

========================================================================================================================
Grid Search Progress: 12/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.1, nhits_coef=0.001
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.084720 | Val Loss: 0.015830
Epoch 21/180 - Train Loss: 0.002768 | Val Loss: 0.006690
Epoch 41/180 - Train Loss: 0.002493 | Val Loss: 0.006130
Epoch 61/180 - Train Loss: 0.002411 | Val Loss: 0.006169
Epoch 81/180 - Train Loss: 0.002384 | Val Loss: 0.005870
Early stop at epoch 84 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.1, nhits=0.001]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 280.266693, MAE: 6.455279, RMSE: 16.741168, R2: 0.994292
Step 2 - MSE: 477.349457, MAE: 8.852373, RMSE: 21.848328, R2: 0.990272
Step 3 - MSE: 660.271667, MAE: 10.472775, RMSE: 25.695752, R2: 0.986533
Step 4 - MSE: 834.964111, MAE: 12.031887, RMSE: 28.895746, R2: 0.982956
Step 5 - MSE: 1022.191162, MAE: 13.618988, RMSE: 31.971724, R2: 0.979117
Step 6 - MSE: 1213.495239, MAE: 15.387777, RMSE: 34.835259, R2: 0.975190
Step 7 - MSE: 1367.649292, MAE: 16.539040, RMSE: 36.981743, R2: 0.972017
Step 8 - MSE: 1519.498779, MAE: 17.470665, RMSE: 38.980749, R2: 0.968885
Step 9 - MSE: 1685.238525, MAE: 18.669138, RMSE: 41.051657, R2: 0.965461
Step 10 - MSE: 1845.701050, MAE: 19.761095, RMSE: 42.961623, R2: 0.962139
Step 11 - MSE: 2011.479370, MAE: 20.865396, RMSE: 44.849519, R2: 0.958703
Step 12 - MSE: 2180.056152, MAE: 22.058971, RMSE: 46.691071, R2: 0.955204

================================================================================
[auto=0.1, nhits=0.001] Overall
================================================================================
Advanced Model - MSE: 1258.179077, MAE: 15.181958, RMSE: 35.470820, R2: 0.974231

save to: results_DENet\pred_auto_0p1_nhits_0p001.csv

Current combination summary:
auto_coef=0.1, nhits_coef=0.001, best_val_loss=0.005817, MSE=1258.179077, MAE=15.181958, RMSE=35.470820, R2=0.974231

========================================================================================================================
Grid Search Progress: 13/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.1, nhits_coef=0.01
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.126847 | Val Loss: 0.014636
Epoch 21/180 - Train Loss: 0.002786 | Val Loss: 0.007390
Epoch 41/180 - Train Loss: 0.002536 | Val Loss: 0.006158
Epoch 61/180 - Train Loss: 0.002438 | Val Loss: 0.005991
Epoch 81/180 - Train Loss: 0.002399 | Val Loss: 0.005960
Epoch 101/180 - Train Loss: 0.002364 | Val Loss: 0.005796
Epoch 121/180 - Train Loss: 0.002358 | Val Loss: 0.005900
Epoch 141/180 - Train Loss: 0.002350 | Val Loss: 0.005789
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))
Early stop at epoch 152 (val loss no improvement for 12 epochs)

================================================================================
[auto=0.1, nhits=0.01]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 283.520813, MAE: 6.364055, RMSE: 16.838076, R2: 0.994226
Step 2 - MSE: 479.840607, MAE: 8.619050, RMSE: 21.905264, R2: 0.990221
Step 3 - MSE: 659.231812, MAE: 10.367240, RMSE: 25.675510, R2: 0.986554
Step 4 - MSE: 837.092163, MAE: 11.945492, RMSE: 28.932545, R2: 0.982913
Step 5 - MSE: 1021.500427, MAE: 13.544340, RMSE: 31.960920, R2: 0.979131
Step 6 - MSE: 1193.888550, MAE: 14.955256, RMSE: 34.552692, R2: 0.975591
Step 7 - MSE: 1360.537720, MAE: 16.245224, RMSE: 36.885468, R2: 0.972162
Step 8 - MSE: 1518.239014, MAE: 17.480291, RMSE: 38.964587, R2: 0.968910
Step 9 - MSE: 1680.987427, MAE: 18.689812, RMSE: 40.999847, R2: 0.965548
Step 10 - MSE: 1839.663086, MAE: 19.831093, RMSE: 42.891294, R2: 0.962263
Step 11 - MSE: 1997.184448, MAE: 20.817383, RMSE: 44.689870, R2: 0.958997
Step 12 - MSE: 2172.930908, MAE: 21.931551, RMSE: 46.614707, R2: 0.955350

================================================================================
[auto=0.1, nhits=0.01] Overall
================================================================================
Advanced Model - MSE: 1253.718384, MAE: 15.065906, RMSE: 35.407886, R2: 0.974322

save to: results_DENet\pred_auto_0p1_nhits_0p01.csv

Current combination summary:
auto_coef=0.1, nhits_coef=0.01, best_val_loss=0.005776, MSE=1253.718384, MAE=15.065906, RMSE=35.407886, R2=0.974322

========================================================================================================================
Grid Search Progress: 14/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.1, nhits_coef=0.05
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.100136 | Val Loss: 0.017549
Epoch 21/180 - Train Loss: 0.002763 | Val Loss: 0.006821
Epoch 41/180 - Train Loss: 0.002492 | Val Loss: 0.006031
Epoch 61/180 - Train Loss: 0.002430 | Val Loss: 0.005967
Epoch 81/180 - Train Loss: 0.002397 | Val Loss: 0.005835
Epoch 101/180 - Train Loss: 0.002384 | Val Loss: 0.005806
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))
Early stop at epoch 114 (val loss no improvement for 12 epochs)

================================================================================
[auto=0.1, nhits=0.05]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 279.133545, MAE: 6.153583, RMSE: 16.707290, R2: 0.994315
Step 2 - MSE: 473.567383, MAE: 8.484457, RMSE: 21.761603, R2: 0.990349
Step 3 - MSE: 655.216187, MAE: 10.607846, RMSE: 25.597191, R2: 0.986636
Step 4 - MSE: 844.573425, MAE: 12.109585, RMSE: 29.061545, R2: 0.982760
Step 5 - MSE: 1020.337036, MAE: 13.523072, RMSE: 31.942715, R2: 0.979155
Step 6 - MSE: 1195.386230, MAE: 15.097088, RMSE: 34.574358, R2: 0.975560
Step 7 - MSE: 1361.521118, MAE: 16.240446, RMSE: 36.898796, R2: 0.972142
Step 8 - MSE: 1515.197021, MAE: 17.494843, RMSE: 38.925532, R2: 0.968973
Step 9 - MSE: 1690.301147, MAE: 18.703096, RMSE: 41.113272, R2: 0.965357
Step 10 - MSE: 1842.517700, MAE: 19.812531, RMSE: 42.924558, R2: 0.962204
Step 11 - MSE: 2005.621094, MAE: 20.827707, RMSE: 44.784161, R2: 0.958823
Step 12 - MSE: 2169.708252, MAE: 21.908825, RMSE: 46.580127, R2: 0.955416

================================================================================
[auto=0.1, nhits=0.05] Overall
================================================================================
Advanced Model - MSE: 1254.421875, MAE: 15.080264, RMSE: 35.417819, R2: 0.974308

save to: results_DENet\pred_auto_0p1_nhits_0p05.csv

Current combination summary:
auto_coef=0.1, nhits_coef=0.05, best_val_loss=0.005772, MSE=1254.421875, MAE=15.080264, RMSE=35.417819, R2=0.974308

========================================================================================================================
Grid Search Progress: 15/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.1, nhits_coef=0.1
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.099230 | Val Loss: 0.015744
Epoch 21/180 - Train Loss: 0.002734 | Val Loss: 0.006625
Epoch 41/180 - Train Loss: 0.002502 | Val Loss: 0.006040
Epoch 61/180 - Train Loss: 0.002421 | Val Loss: 0.006052
Epoch 81/180 - Train Loss: 0.002387 | Val Loss: 0.005826
Epoch 101/180 - Train Loss: 0.002362 | Val Loss: 0.005793
Early stop at epoch 120 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.1, nhits=0.1]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 279.104248, MAE: 6.183666, RMSE: 16.706413, R2: 0.994316
Step 2 - MSE: 472.458527, MAE: 8.447832, RMSE: 21.736111, R2: 0.990371
Step 3 - MSE: 655.840759, MAE: 10.325613, RMSE: 25.609388, R2: 0.986623
Step 4 - MSE: 837.835022, MAE: 11.976472, RMSE: 28.945380, R2: 0.982898
Step 5 - MSE: 1019.094360, MAE: 13.505966, RMSE: 31.923257, R2: 0.979181
Step 6 - MSE: 1195.273926, MAE: 14.955920, RMSE: 34.572734, R2: 0.975563
Step 7 - MSE: 1359.364868, MAE: 16.377504, RMSE: 36.869566, R2: 0.972186
Step 8 - MSE: 1516.595581, MAE: 17.436663, RMSE: 38.943492, R2: 0.968944
Step 9 - MSE: 1684.197266, MAE: 18.610357, RMSE: 41.038973, R2: 0.965482
Step 10 - MSE: 1846.216064, MAE: 19.743757, RMSE: 42.967616, R2: 0.962129
Step 11 - MSE: 2006.691406, MAE: 20.751947, RMSE: 44.796109, R2: 0.958801
Step 12 - MSE: 2171.656982, MAE: 21.949257, RMSE: 46.601041, R2: 0.955376

================================================================================
[auto=0.1, nhits=0.1] Overall
================================================================================
Advanced Model - MSE: 1253.692871, MAE: 15.022086, RMSE: 35.407526, R2: 0.974322

save to: results_DENet\pred_auto_0p1_nhits_0p1.csv

Current combination summary:
auto_coef=0.1, nhits_coef=0.1, best_val_loss=0.005768, MSE=1253.692871, MAE=15.022086, RMSE=35.407526, R2=0.974322

========================================================================================================================
Grid Search Progress: 16/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.2, nhits_coef=0.0
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.115916 | Val Loss: 0.028996
Epoch 21/180 - Train Loss: 0.002747 | Val Loss: 0.006661
Epoch 41/180 - Train Loss: 0.002524 | Val Loss: 0.006694
Epoch 61/180 - Train Loss: 0.002434 | Val Loss: 0.005898
Early stop at epoch 77 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.2, nhits=0.0]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 280.307404, MAE: 6.235885, RMSE: 16.742383, R2: 0.994291
Step 2 - MSE: 486.085876, MAE: 9.065719, RMSE: 22.047355, R2: 0.990094
Step 3 - MSE: 657.906250, MAE: 11.130317, RMSE: 25.649683, R2: 0.986581
Step 4 - MSE: 840.224243, MAE: 13.207478, RMSE: 28.986622, R2: 0.982849
Step 5 - MSE: 1017.685974, MAE: 14.061630, RMSE: 31.901191, R2: 0.979209
Step 6 - MSE: 1199.451050, MAE: 15.433993, RMSE: 34.633092, R2: 0.975477
Step 7 - MSE: 1364.882935, MAE: 16.540817, RMSE: 36.944322, R2: 0.972073
Step 8 - MSE: 1512.340332, MAE: 17.798643, RMSE: 38.888820, R2: 0.969031
Step 9 - MSE: 1683.412354, MAE: 18.950100, RMSE: 41.029408, R2: 0.965498
Step 10 - MSE: 1846.184448, MAE: 19.772762, RMSE: 42.967249, R2: 0.962129
Step 11 - MSE: 2005.947144, MAE: 21.070486, RMSE: 44.787801, R2: 0.958817
Step 12 - MSE: 2176.849609, MAE: 21.949924, RMSE: 46.656721, R2: 0.955270

================================================================================
[auto=0.2, nhits=0.0] Overall
================================================================================
Advanced Model - MSE: 1255.938843, MAE: 15.434811, RMSE: 35.439227, R2: 0.974277

save to: results_DENet\pred_auto_0p2_nhits_0p0.csv

Current combination summary:
auto_coef=0.2, nhits_coef=0.0, best_val_loss=0.005839, MSE=1255.938843, MAE=15.434811, RMSE=35.439227, R2=0.974277

========================================================================================================================
Grid Search Progress: 17/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.2, nhits_coef=0.001
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.049433 | Val Loss: 0.013656
Epoch 21/180 - Train Loss: 0.002697 | Val Loss: 0.006612
Epoch 41/180 - Train Loss: 0.002470 | Val Loss: 0.006086
Epoch 61/180 - Train Loss: 0.002434 | Val Loss: 0.005859
Epoch 81/180 - Train Loss: 0.002385 | Val Loss: 0.005788
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))
Early stop at epoch 98 (val loss no improvement for 12 epochs)

================================================================================
[auto=0.2, nhits=0.001]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 277.345367, MAE: 6.218482, RMSE: 16.653689, R2: 0.994352
Step 2 - MSE: 465.634613, MAE: 8.354451, RMSE: 21.578568, R2: 0.990510
Step 3 - MSE: 653.112366, MAE: 10.410565, RMSE: 25.556063, R2: 0.986679
Step 4 - MSE: 833.892700, MAE: 12.045851, RMSE: 28.877200, R2: 0.982978
Step 5 - MSE: 1010.288330, MAE: 13.727667, RMSE: 31.785033, R2: 0.979360
Step 6 - MSE: 1194.472900, MAE: 15.774774, RMSE: 34.561147, R2: 0.975579
Step 7 - MSE: 1357.066650, MAE: 16.342474, RMSE: 36.838386, R2: 0.972233
Step 8 - MSE: 1505.328979, MAE: 17.590412, RMSE: 38.798569, R2: 0.969175
Step 9 - MSE: 1672.837646, MAE: 18.803015, RMSE: 40.900338, R2: 0.965715
Step 10 - MSE: 1841.510132, MAE: 19.796522, RMSE: 42.912820, R2: 0.962225
Step 11 - MSE: 1990.064331, MAE: 20.904318, RMSE: 44.610137, R2: 0.959143
Step 12 - MSE: 2157.828613, MAE: 22.342209, RMSE: 46.452434, R2: 0.955660

================================================================================
[auto=0.2, nhits=0.001] Overall
================================================================================
Advanced Model - MSE: 1246.615234, MAE: 15.192571, RMSE: 35.307439, R2: 0.974467

save to: results_DENet\pred_auto_0p2_nhits_0p001.csv

Current combination summary:
auto_coef=0.2, nhits_coef=0.001, best_val_loss=0.005754, MSE=1246.615234, MAE=15.192571, RMSE=35.307439, R2=0.974467

========================================================================================================================
Grid Search Progress: 18/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.2, nhits_coef=0.01
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.043159 | Val Loss: 0.013080
Epoch 21/180 - Train Loss: 0.002617 | Val Loss: 0.006437
Epoch 41/180 - Train Loss: 0.002463 | Val Loss: 0.005944
Epoch 61/180 - Train Loss: 0.002416 | Val Loss: 0.005812
Epoch 81/180 - Train Loss: 0.002379 | Val Loss: 0.005762
Epoch 101/180 - Train Loss: 0.002362 | Val Loss: 0.005755
Early stop at epoch 117 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.2, nhits=0.01]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 272.233856, MAE: 6.105701, RMSE: 16.499511, R2: 0.994456
Step 2 - MSE: 468.427856, MAE: 8.402270, RMSE: 21.643194, R2: 0.990453
Step 3 - MSE: 647.023621, MAE: 10.171151, RMSE: 25.436659, R2: 0.986803
Step 4 - MSE: 830.290222, MAE: 11.893506, RMSE: 28.814757, R2: 0.983052
Step 5 - MSE: 1016.826538, MAE: 13.549880, RMSE: 31.887718, R2: 0.979227
Step 6 - MSE: 1192.380127, MAE: 14.991706, RMSE: 34.530858, R2: 0.975622
Step 7 - MSE: 1354.854736, MAE: 16.229410, RMSE: 36.808351, R2: 0.972278
Step 8 - MSE: 1514.799683, MAE: 17.397995, RMSE: 38.920428, R2: 0.968981
Step 9 - MSE: 1682.172485, MAE: 18.585100, RMSE: 41.014296, R2: 0.965524
Step 10 - MSE: 1841.327759, MAE: 19.730389, RMSE: 42.910695, R2: 0.962229
Step 11 - MSE: 1999.840088, MAE: 20.816071, RMSE: 44.719572, R2: 0.958942
Step 12 - MSE: 2172.833252, MAE: 21.900850, RMSE: 46.613659, R2: 0.955352

================================================================================
[auto=0.2, nhits=0.01] Overall
================================================================================
Advanced Model - MSE: 1249.416870, MAE: 14.981174, RMSE: 35.347091, R2: 0.974410

save to: results_DENet\pred_auto_0p2_nhits_0p01.csv

Current combination summary:
auto_coef=0.2, nhits_coef=0.01, best_val_loss=0.005743, MSE=1249.416870, MAE=14.981174, RMSE=35.347091, R2=0.974410

========================================================================================================================
Grid Search Progress: 19/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.2, nhits_coef=0.05
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.113340 | Val Loss: 0.015680
Epoch 21/180 - Train Loss: 0.002744 | Val Loss: 0.006633
Epoch 41/180 - Train Loss: 0.002475 | Val Loss: 0.007315
Epoch 61/180 - Train Loss: 0.002409 | Val Loss: 0.005838
Epoch 81/180 - Train Loss: 0.002379 | Val Loss: 0.005807
Epoch 101/180 - Train Loss: 0.002369 | Val Loss: 0.005772
Early stop at epoch 102 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.2, nhits=0.05]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 276.232361, MAE: 6.324336, RMSE: 16.620239, R2: 0.994374
Step 2 - MSE: 476.332092, MAE: 8.683879, RMSE: 21.825034, R2: 0.990292
Step 3 - MSE: 654.403503, MAE: 10.404208, RMSE: 25.581312, R2: 0.986652
Step 4 - MSE: 838.589539, MAE: 12.041463, RMSE: 28.958410, R2: 0.982882
Step 5 - MSE: 1017.828918, MAE: 13.577352, RMSE: 31.903431, R2: 0.979206
Step 6 - MSE: 1199.507324, MAE: 15.070195, RMSE: 34.633904, R2: 0.975476
Step 7 - MSE: 1357.479370, MAE: 16.305273, RMSE: 36.843987, R2: 0.972225
Step 8 - MSE: 1509.936890, MAE: 17.512077, RMSE: 38.857906, R2: 0.969080
Step 9 - MSE: 1676.241333, MAE: 18.604107, RMSE: 40.941926, R2: 0.965645
Step 10 - MSE: 1831.426025, MAE: 19.743486, RMSE: 42.795164, R2: 0.962432
Step 11 - MSE: 1999.742188, MAE: 20.812334, RMSE: 44.718477, R2: 0.958944
Step 12 - MSE: 2168.303711, MAE: 22.019634, RMSE: 46.565048, R2: 0.955445

================================================================================
[auto=0.2, nhits=0.05] Overall
================================================================================
Advanced Model - MSE: 1250.501953, MAE: 15.091530, RMSE: 35.362437, R2: 0.974388

save to: results_DENet\pred_auto_0p2_nhits_0p05.csv

Current combination summary:
auto_coef=0.2, nhits_coef=0.05, best_val_loss=0.005762, MSE=1250.501953, MAE=15.091530, RMSE=35.362437, R2=0.974388

========================================================================================================================
Grid Search Progress: 20/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.2, nhits_coef=0.1
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.100602 | Val Loss: 0.019972
Epoch 21/180 - Train Loss: 0.002760 | Val Loss: 0.006689
Epoch 41/180 - Train Loss: 0.002515 | Val Loss: 0.006289
Epoch 61/180 - Train Loss: 0.002427 | Val Loss: 0.005961
Epoch 81/180 - Train Loss: 0.002387 | Val Loss: 0.005840
Epoch 101/180 - Train Loss: 0.002349 | Val Loss: 0.005853
Early stop at epoch 102 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.2, nhits=0.1]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 277.597961, MAE: 6.393969, RMSE: 16.661271, R2: 0.994347
Step 2 - MSE: 475.510834, MAE: 8.704639, RMSE: 21.806211, R2: 0.990309
Step 3 - MSE: 655.492615, MAE: 10.464749, RMSE: 25.602590, R2: 0.986630
Step 4 - MSE: 841.206726, MAE: 12.254363, RMSE: 29.003564, R2: 0.982829
Step 5 - MSE: 1021.731628, MAE: 13.760094, RMSE: 31.964537, R2: 0.979127
Step 6 - MSE: 1202.158081, MAE: 15.230990, RMSE: 34.672151, R2: 0.975422
Step 7 - MSE: 1363.720459, MAE: 16.386229, RMSE: 36.928586, R2: 0.972097
Step 8 - MSE: 1519.228882, MAE: 17.563911, RMSE: 38.977287, R2: 0.968890
Step 9 - MSE: 1687.460327, MAE: 18.724302, RMSE: 41.078709, R2: 0.965415
Step 10 - MSE: 1841.575684, MAE: 19.840204, RMSE: 42.913584, R2: 0.962224
Step 11 - MSE: 2011.814209, MAE: 21.005657, RMSE: 44.853252, R2: 0.958696
Step 12 - MSE: 2174.946777, MAE: 21.933550, RMSE: 46.636325, R2: 0.955309

================================================================================
[auto=0.2, nhits=0.1] Overall
================================================================================
Advanced Model - MSE: 1256.035767, MAE: 15.188549, RMSE: 35.440595, R2: 0.974275

save to: results_DENet\pred_auto_0p2_nhits_0p1.csv

Current combination summary:
auto_coef=0.2, nhits_coef=0.1, best_val_loss=0.005806, MSE=1256.035767, MAE=15.188549, RMSE=35.440595, R2=0.974275

========================================================================================================================
Grid Search Progress: 21/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.5, nhits_coef=0.0
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.099731 | Val Loss: 0.021885
Epoch 21/180 - Train Loss: 0.002729 | Val Loss: 0.006933
Epoch 41/180 - Train Loss: 0.002497 | Val Loss: 0.005944
Epoch 61/180 - Train Loss: 0.002403 | Val Loss: 0.005941
Epoch 81/180 - Train Loss: 0.002374 | Val Loss: 0.005756
Epoch 101/180 - Train Loss: 0.002350 | Val Loss: 0.005722
Early stop at epoch 119 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.5, nhits=0.0]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 259.718994, MAE: 6.291327, RMSE: 16.115800, R2: 0.994711
Step 2 - MSE: 455.960602, MAE: 8.441657, RMSE: 21.353234, R2: 0.990708
Step 3 - MSE: 635.725891, MAE: 10.144108, RMSE: 25.213605, R2: 0.987033
Step 4 - MSE: 818.433411, MAE: 11.944576, RMSE: 28.608275, R2: 0.983294
Step 5 - MSE: 1001.104919, MAE: 13.512115, RMSE: 31.640242, R2: 0.979548
Step 6 - MSE: 1185.463013, MAE: 14.947948, RMSE: 34.430553, R2: 0.975763
Step 7 - MSE: 1348.675049, MAE: 16.185102, RMSE: 36.724311, R2: 0.972405
Step 8 - MSE: 1504.752686, MAE: 17.386703, RMSE: 38.791142, R2: 0.969187
Step 9 - MSE: 1669.391602, MAE: 18.769926, RMSE: 40.858189, R2: 0.965786
Step 10 - MSE: 1822.086792, MAE: 19.947582, RMSE: 42.685909, R2: 0.962624
Step 11 - MSE: 1994.517578, MAE: 20.727203, RMSE: 44.660022, R2: 0.959051
Step 12 - MSE: 2151.064697, MAE: 22.330044, RMSE: 46.379572, R2: 0.955799

================================================================================
[auto=0.5, nhits=0.0] Overall
================================================================================
Advanced Model - MSE: 1237.240356, MAE: 15.052341, RMSE: 35.174428, R2: 0.974659

save to: results_DENet\pred_auto_0p5_nhits_0p0.csv

Current combination summary:
auto_coef=0.5, nhits_coef=0.0, best_val_loss=0.005673, MSE=1237.240356, MAE=15.052341, RMSE=35.174428, R2=0.974659

========================================================================================================================
Grid Search Progress: 22/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.5, nhits_coef=0.001
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.084782 | Val Loss: 0.021112
Epoch 21/180 - Train Loss: 0.002737 | Val Loss: 0.006889
Epoch 41/180 - Train Loss: 0.002488 | Val Loss: 0.006789
Epoch 61/180 - Train Loss: 0.002425 | Val Loss: 0.006029
Epoch 81/180 - Train Loss: 0.002368 | Val Loss: 0.005706
Early stop at epoch 92 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.5, nhits=0.001]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 263.339294, MAE: 6.194339, RMSE: 16.227732, R2: 0.994637
Step 2 - MSE: 459.695312, MAE: 8.482120, RMSE: 21.440506, R2: 0.990631
Step 3 - MSE: 635.367310, MAE: 10.147050, RMSE: 25.206493, R2: 0.987041
Step 4 - MSE: 821.081177, MAE: 12.597677, RMSE: 28.654514, R2: 0.983240
Step 5 - MSE: 1000.079590, MAE: 13.555561, RMSE: 31.624035, R2: 0.979569
Step 6 - MSE: 1182.818237, MAE: 15.066466, RMSE: 34.392125, R2: 0.975817
Step 7 - MSE: 1348.182983, MAE: 16.275045, RMSE: 36.717611, R2: 0.972415
Step 8 - MSE: 1507.875122, MAE: 17.494383, RMSE: 38.831368, R2: 0.969123
Step 9 - MSE: 1677.456543, MAE: 18.657253, RMSE: 40.956764, R2: 0.965620
Step 10 - MSE: 1832.643188, MAE: 19.741407, RMSE: 42.809382, R2: 0.962407
Step 11 - MSE: 1992.817017, MAE: 21.080030, RMSE: 44.640979, R2: 0.959086
Step 12 - MSE: 2160.545410, MAE: 21.897215, RMSE: 46.481667, R2: 0.955605

================================================================================
[auto=0.5, nhits=0.001] Overall
================================================================================
Advanced Model - MSE: 1240.158569, MAE: 15.099051, RMSE: 35.215885, R2: 0.974599

save to: results_DENet\pred_auto_0p5_nhits_0p001.csv

Current combination summary:
auto_coef=0.5, nhits_coef=0.001, best_val_loss=0.005701, MSE=1240.158569, MAE=15.099051, RMSE=35.215885, R2=0.974599

========================================================================================================================
Grid Search Progress: 23/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.5, nhits_coef=0.01
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.164976 | Val Loss: 0.021242
Epoch 21/180 - Train Loss: 0.002752 | Val Loss: 0.006569
Epoch 41/180 - Train Loss: 0.002561 | Val Loss: 0.006822
Epoch 61/180 - Train Loss: 0.002434 | Val Loss: 0.005853
Epoch 81/180 - Train Loss: 0.002404 | Val Loss: 0.005826
Epoch 101/180 - Train Loss: 0.002347 | Val Loss: 0.005798
Early stop at epoch 101 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.5, nhits=0.01]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 268.101501, MAE: 6.529120, RMSE: 16.373805, R2: 0.994540
Step 2 - MSE: 459.927155, MAE: 8.561670, RMSE: 21.445912, R2: 0.990627
Step 3 - MSE: 641.963135, MAE: 10.304820, RMSE: 25.336991, R2: 0.986906
Step 4 - MSE: 821.059814, MAE: 12.113028, RMSE: 28.654141, R2: 0.983240
Step 5 - MSE: 1001.493652, MAE: 13.445437, RMSE: 31.646385, R2: 0.979540
Step 6 - MSE: 1184.438721, MAE: 15.231281, RMSE: 34.415676, R2: 0.975784
Step 7 - MSE: 1352.001343, MAE: 16.353006, RMSE: 36.769571, R2: 0.972337
Step 8 - MSE: 1512.997070, MAE: 17.695387, RMSE: 38.897263, R2: 0.969018
Step 9 - MSE: 1671.874146, MAE: 18.576754, RMSE: 40.888558, R2: 0.965735
Step 10 - MSE: 1829.666382, MAE: 19.751574, RMSE: 42.774600, R2: 0.962468
Step 11 - MSE: 1993.778687, MAE: 20.811876, RMSE: 44.651749, R2: 0.959066
Step 12 - MSE: 2156.992920, MAE: 22.035803, RMSE: 46.443438, R2: 0.955678

================================================================================
[auto=0.5, nhits=0.01] Overall
================================================================================
Advanced Model - MSE: 1241.190552, MAE: 15.117467, RMSE: 35.230534, R2: 0.974578

save to: results_DENet\pred_auto_0p5_nhits_0p01.csv

Current combination summary:
auto_coef=0.5, nhits_coef=0.01, best_val_loss=0.005699, MSE=1241.190552, MAE=15.117467, RMSE=35.230534, R2=0.974578

========================================================================================================================
Grid Search Progress: 24/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.5, nhits_coef=0.05
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.137365 | Val Loss: 0.025248
Epoch 21/180 - Train Loss: 0.002701 | Val Loss: 0.006520
Epoch 41/180 - Train Loss: 0.002545 | Val Loss: 0.006117
Epoch 61/180 - Train Loss: 0.002460 | Val Loss: 0.006412
Epoch 81/180 - Train Loss: 0.002443 | Val Loss: 0.006080
Early stop at epoch 95 (val loss no improvement for 12 epochs)
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))

================================================================================
[auto=0.5, nhits=0.05]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 289.359009, MAE: 7.459186, RMSE: 17.010556, R2: 0.994107
Step 2 - MSE: 477.848907, MAE: 8.767773, RMSE: 21.859755, R2: 0.990261
Step 3 - MSE: 670.553772, MAE: 10.834233, RMSE: 25.895053, R2: 0.986323
Step 4 - MSE: 850.350647, MAE: 12.357096, RMSE: 29.160772, R2: 0.982642
Step 5 - MSE: 1025.226562, MAE: 13.736050, RMSE: 32.019159, R2: 0.979055
Step 6 - MSE: 1208.955688, MAE: 15.264461, RMSE: 34.770040, R2: 0.975283
Step 7 - MSE: 1366.384399, MAE: 16.389265, RMSE: 36.964637, R2: 0.972042
Step 8 - MSE: 1521.784180, MAE: 18.159948, RMSE: 39.010052, R2: 0.968838
Step 9 - MSE: 1698.388306, MAE: 19.375198, RMSE: 41.211507, R2: 0.965191
Step 10 - MSE: 1851.401123, MAE: 20.661709, RMSE: 43.027911, R2: 0.962022
Step 11 - MSE: 2017.530518, MAE: 21.254915, RMSE: 44.916929, R2: 0.958579
Step 12 - MSE: 2191.592773, MAE: 22.302523, RMSE: 46.814450, R2: 0.954967

================================================================================
[auto=0.5, nhits=0.05] Overall
================================================================================
Advanced Model - MSE: 1264.114380, MAE: 15.546870, RMSE: 35.554386, R2: 0.974109

save to: results_DENet\pred_auto_0p5_nhits_0p05.csv

Current combination summary:
auto_coef=0.5, nhits_coef=0.05, best_val_loss=0.005945, MSE=1264.114380, MAE=15.546870, RMSE=35.554386, R2=0.974109

========================================================================================================================
Grid Search Progress: 25/25
========================================================================================================================

####################################################################################################
Current combination -> auto_coef=0.5, nhits_coef=0.1
####################################################################################################
length (look_back): 20
Data preprocessing completed - Training set batches: 244, Test set batches: 61
finished: 1,644
Epoch 1/180 - Train Loss: 0.119642 | Val Loss: 0.023939
Epoch 21/180 - Train Loss: 0.002662 | Val Loss: 0.006475
Epoch 41/180 - Train Loss: 0.002490 | Val Loss: 0.006055
Epoch 61/180 - Train Loss: 0.002392 | Val Loss: 0.005840
Epoch 81/180 - Train Loss: 0.002356 | Val Loss: 0.005801
D:\桌面\FanGao\Final_Experiment\WR\min_ablation\Wangge.py:361: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path, map_location=device))
Early stop at epoch 94 (val loss no improvement for 12 epochs)

================================================================================
[auto=0.5, nhits=0.1]  (Step 1 ~ Step 12)
================================================================================
Step 1 - MSE: 273.492401, MAE: 6.427969, RMSE: 16.537606, R2: 0.994430
Step 2 - MSE: 467.535522, MAE: 8.885637, RMSE: 21.622570, R2: 0.990472
Step 3 - MSE: 646.270142, MAE: 10.592638, RMSE: 25.421844, R2: 0.986818
Step 4 - MSE: 837.534668, MAE: 12.246963, RMSE: 28.940191, R2: 0.982904
Step 5 - MSE: 1013.686401, MAE: 13.682988, RMSE: 31.838442, R2: 0.979291
Step 6 - MSE: 1192.279419, MAE: 15.194761, RMSE: 34.529399, R2: 0.975624
Step 7 - MSE: 1357.638428, MAE: 16.683842, RMSE: 36.846145, R2: 0.972221
Step 8 - MSE: 1513.900024, MAE: 17.729994, RMSE: 38.908868, R2: 0.968999
Step 9 - MSE: 1692.126831, MAE: 18.928432, RMSE: 41.135469, R2: 0.965320
Step 10 - MSE: 1846.568115, MAE: 19.967999, RMSE: 42.971713, R2: 0.962121
Step 11 - MSE: 2010.101685, MAE: 20.958200, RMSE: 44.834158, R2: 0.958731
Step 12 - MSE: 2174.109619, MAE: 22.045866, RMSE: 46.627348, R2: 0.955326

================================================================================
[auto=0.5, nhits=0.1] Overall
================================================================================
Advanced Model - MSE: 1252.103638, MAE: 15.278779, RMSE: 35.385076, R2: 0.974355

save to: results_DENet\pred_auto_0p5_nhits_0p1.csv

Current combination summary:
auto_coef=0.5, nhits_coef=0.1, best_val_loss=0.005785, MSE=1252.103638, MAE=15.278779, RMSE=35.385076, R2=0.974355

Grid search summary saved to: results_DENet\grid_search_results.csv

########################################################################################################################
Grid Search Final Ranking (Top 10)
########################################################################################################################
 auto_coef  nhits_coef  best_val_loss         mse       mae      rmse       r2
      0.50       0.000       0.005673 1237.240356 15.052341 35.174428 0.974659
      0.50       0.001       0.005701 1240.158569 15.099051 35.215885 0.974599
      0.50       0.010       0.005699 1241.190552 15.117467 35.230534 0.974578
      0.00       0.001       0.005692 1245.424683 14.956008 35.290575 0.974492
      0.20       0.001       0.005754 1246.615234 15.192571 35.307439 0.974467
      0.00       0.050       0.005720 1247.489258 15.000566 35.319814 0.974449
      0.05       0.100       0.005711 1247.615845 14.960414 35.321606 0.974447
      0.00       0.100       0.005707 1247.709473 14.967637 35.322931 0.974445
      0.00       0.010       0.005717 1248.825317 14.966914 35.338723 0.974422
      0.20       0.010       0.005743 1249.416870 14.981174 35.347091 0.974410

########################################################################################################################
Best Combination
########################################################################################################################
auto_coef=0.5, nhits_coef=0.0, best_val_loss=0.005673, MSE=1237.240356, MAE=15.052341, RMSE=35.174428, R2=0.974659
Best combination info saved to: results_DENet\best_combination.txt

finished

进程已结束，退出代码为 0


"""