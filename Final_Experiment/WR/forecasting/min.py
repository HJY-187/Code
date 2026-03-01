import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # （open/high/low/close/volume/open_interest）
        x = self.data[idx:idx + self.seq_len, :6]
        # predict target
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, 3]
        # close
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
        df_work[col] = np.log(df_work[col] + 1e-8)  # log(0)

    # ATR
    df_work['TR'] = np.maximum(
        df_work['high'] - df_work['low'],
        np.maximum(
            np.abs(df_work['high'] - df_work['close'].shift(1)),
            np.abs(df_work['low'] - df_work['close'].shift(1))
        )
    )
    df_work['ATR'] = df_work['TR'].rolling(window=14).mean()

    # RSI
    delta = df_work['close'] - df_work['close'].shift(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().replace(0, 1e-8) 
    rs = avg_gain / avg_loss
    df_work['RSI'] = 100 - (100 / (1 + rs))

    # roll_return
    df_work['roll_return'] = (df_work['close'] - df_work['close'].shift(5)) / df_work['close'].shift(5)

    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # features
    features = df_work[[
        "open", "high", "low", "close", "volume", "open_interest",
        "ATR", "RSI", "roll_return"
    ]].values

    # daily
    dates_work = df_work["datetime"].values

    # 8:2
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back  

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]

    # RobustScaler
    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    train_ds = TimeSeriesDataset(train_data, seq_len=look_back, pred_len=pred_len)
    test_ds = TimeSeriesDataset(test_data, seq_len=look_back, pred_len=pred_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)  
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, test_loader, scaler, dates_work, train_size, look_back


# =============================== model===============================
class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels,
                                   bias=False)
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
        """moving avg"""
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
        self.net = nn.Sequential(
            nn.Linear(seq_len, seq_len // hidden_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len // hidden_ratio, pred_len)
        )

    def forward(self, x):  # x: [B,S] (close)
        return self.net(self.norm(x))


# =============================== Main Path ===============================
class DENet(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, features: int = 6,
                 kernel_size: int = 5, use_nhits: bool = True, use_auto: bool = True,
                 dropout: float = 0.3, nhits_hidden_ratio: int = 4,
                 nhits_init: float = 0.001, auto_init: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.base_feat_num = features
        self.kernel_size = kernel_size
        self.use_nhits = use_nhits
        self.use_auto = use_auto

        # Learnable Fusion Weights (Optimized Init)
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

        if self.use_auto:
            self.auto_block = AuxiliaryStreams1(channels=features, seq_len=seq_len, pred_len=pred_len)
        if self.use_nhits:
            self.nhits_block = AuxiliaryStreams2(seq_len=seq_len, pred_len=pred_len,
                                              dropout=dropout, hidden_ratio=nhits_hidden_ratio)

        self.h_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def moving_avg(self, x, kernel_size):
        padding = (kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):  # [B, seq_len, features]
        # base features
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

        # Ensure shapes align
        last_close = last_close.view(-1, 1)

        true_diff = true - last_close
        pred_diff = pred - last_close

        # Penalty is positive if signs differ (product is negative)
        penalty = F.relu(-1.0 * true_diff * pred_diff)

        return num_loss + self.alpha * torch.mean(penalty)


def denormalize_predictions(scaled_preds, scaler):
    """Inverse Standardization"""
    n, p = scaled_preds.shape
    original = np.zeros_like(scaled_preds)
    for i in range(p):
        dummy = np.zeros((n, 9))  # 9个特征列
        dummy[:, 3] = scaled_preds[:, i]  # close列在第4列（索引3）
        den = scaler.inverse_transform(dummy)
        original[:, i] = np.exp(den[:, 3]) - 1e-8  # 反对数变换
    return original


def train_model(model, train_loader, val_loader, epochs=160, lr=8e-4, model_dir='models_DENet'):
    """Early Stopping and Learning Rate Scheduling"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8, verbose=False)
    crit = DirectionalLoss(alpha=1.0)  # Optimized: Directional Loss

    best_val_loss = float('inf')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f"{model_dir}/DENet_best.pth"
    patience = 0
    early_stop_patience = 12

    for ep in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            # Extract last close for Directional Loss
            last_close = bx[:, -1, 3]

            opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, by, last_close)  # Pass last_close
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                last_close = bx[:, -1, 3]

                pred = model(bx)
                val_loss += crit(pred, by, last_close).item()

        # loss
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))

        # Learning Rate Scheduling
        sch.step(val_loss)

        if ep % 20 == 0:
            print(f"Epoch {ep + 1}/{epochs} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stop at epoch {ep + 1} (val loss no improvement for {early_stop_patience} epochs)")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def generate_and_save_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path):
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

    #Inverse Standardization
    preds = denormalize_predictions(preds_scaled, scaler)
    trues = denormalize_predictions(trues_scaled, scaler)

    res_len = preds.shape[0]
    start_idx = train_size + 2 * look_back
    res_dates = dates_work[start_idx: start_idx + res_len]

    print("\n" + "=" * 80)
    print("(Step 1 ~ Step 12)")
    print("=" * 80)
    for step in range(PRED_LEN):

        true_step = trues[:, step]
        pred_step = preds[:, step]


        mse_step = mean_squared_error(true_step, pred_step)
        mae_step = mean_absolute_error(true_step, pred_step)
        rmse_step = math.sqrt(mse_step)
        r2_step = r2_score(true_step, pred_step)


        print(f"Step {step + 1} - MSE: {mse_step:.6f}, MAE: {mae_step:.6f}, RMSE: {rmse_step:.6f}, R2: {r2_step:.6f}")

    print("\n" + "=" * 80)
    print("=" * 80)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(trues, preds)
    print(f"Advanced Model - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

    # ==========save ==========
    df_out = pd.DataFrame()
    df_out['datetime'] = res_dates
    for k in range(preds.shape[1]):
        df_out[f'pred_step_{k + 1}'] = preds[:, k]
    df_out['pred_avg_1hr'] = preds.mean(axis=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\n save to: {save_path}")


# =============================== Main ===============================
def main(data_file="WR_5.csv", output_dir="results_DENet", model_dir="models_DENet",
         pretrained_path: str | None = None, look_back_override: int | None = None):
    """
    Main function: Data loading -> Preprocessing -> Model training -> Prediction -> Result saving
    Parameters:
    - data_file: Path to the input data file (CSV format)
    - output_dir: Directory for saving output results
    - model_dir: Directory for saving the model
    - pretrained_path: Path to the pretrained model (optional)
    - look_back_override: Length of the input sequence (overrides the default value)
"""
    print(f"device: {device}")
    print(f"file: {data_file}")

    # 1. 加载数据
    try:
        df = pd.read_csv(data_file)
        print(f"Data loaded successfully， {len(df)} ")
    except FileNotFoundError:
        print(f"fail '{data_file}'")
        return
    except Exception as e:
        print(f"fail：{e}")
        return

    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    if df["datetime"].isnull().any():
        print("warning：datetime invalid，delete")
        df = df.dropna(subset=["datetime"])

    lb_def = 48 if "30" in os.path.basename(data_file) else 20
    look_back = look_back_override if look_back_override is not None else lb_def
    print(f"length (look_back): {look_back}")

    print("begin...")
    try:
        train_loader, test_loader, scaler, dates_work, train_size, look_back = preprocess_and_split_data(
            df, look_back, PRED_LEN
        )
        print(f"Data preprocessing completed - Training set batches: {len(train_loader)}, Test set batches: {len(test_loader)}")
    except Exception as e:
        print(f"fail：{e}")
        return

    # 4. Optimized Hyperparameters
    model = DENet(
        seq_len=look_back, pred_len=PRED_LEN, features=6,
        dropout=0.3,  # High dropout for minute noise
        nhits_hidden_ratio=4,  # Reduced capacity
        nhits_init=0.001,  # Conservative init
        auto_init=0.1
    ).to(device)
    print(f"finished: {sum(p.numel() for p in model.parameters()):,}")

    # 加载预训练权重（可选）
    if pretrained_path is not None and os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
            print(f" Successfully loaded pretrained weights: {pretrained_path}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")

    # 5. train
    print("begin...")
    try:
        model = train_model(model, train_loader, test_loader, epochs=180, lr=5e-4, model_dir=model_dir)
    except Exception as e:
        print(f"fail：{e}")
        return

    # 6. save
    print("begin...")
    save_path = os.path.join("..", "strategy", "pred_minute_advanced.csv")
    try:
        generate_and_save_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path)
    except Exception as e:
        print(f"fail：{e}")
        return

    print("\nfinished")
    return None


# =============================== Enter ===============================
if __name__ == "__main__":
    import sys

    # python script.py [data_file] [output_dir] [model_dir] [pretrained_path] [look_back]
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "results_DENet"
        model_dir = sys.argv[3] if len(sys.argv) > 3 else "models_DENet"
        pretrained = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "None" else None
        look_back_arg = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in ("", "None") else None
        main(data_file, output_dir, model_dir, pretrained_path=pretrained, look_back_override=look_back_arg)
    else:
        main()

"""
WR
================================================================================
Step 1 - MSE: 257.661224, MAE: 5.937236, RMSE: 16.051829, R2: 0.994753
Step 2 - MSE: 455.133850, MAE: 8.359082, RMSE: 21.333866, R2: 0.990724
Step 3 - MSE: 636.536743, MAE: 10.516687, RMSE: 25.229680, R2: 0.987017
Step 4 - MSE: 817.282166, MAE: 11.761833, RMSE: 28.588147, R2: 0.983317
Step 5 - MSE: 1003.985718, MAE: 13.481165, RMSE: 31.685734, R2: 0.979489
Step 6 - MSE: 1186.256104, MAE: 14.926065, RMSE: 34.442069, R2: 0.975747
Step 7 - MSE: 1351.431152, MAE: 16.223661, RMSE: 36.761816, R2: 0.972348
Step 8 - MSE: 1503.065186, MAE: 17.515228, RMSE: 38.769385, R2: 0.969221
Step 9 - MSE: 1675.114746, MAE: 18.584747, RMSE: 40.928166, R2: 0.965668
Step 10 - MSE: 1833.474609, MAE: 19.721170, RMSE: 42.819092, R2: 0.962390
Step 11 - MSE: 1991.497559, MAE: 20.860798, RMSE: 44.626198, R2: 0.959113
Step 12 - MSE: 2164.589600, MAE: 21.893227, RMSE: 46.525150, R2: 0.955521
================================================================================
Advanced Model - MSE: 1239.668701, MAE: 14.981746, RMSE: 35.208929, R2: 0.974609


"""