import os
import math
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LEN = 12


# =============================== Model: DLinear ===============================
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearModel(nn.Module):
    def __init__(self, seq_len=48, pred_len=12, enc_in=6):
        super(DLinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.decompsition = SeriesDecomp(kernel_size=25)
        # Linear layer for trend and seasonality
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        seasonal_init, trend_init = self.decompsition(x)

        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output

        x = x.permute(0, 2, 1)

        return x[:, :, 3]


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
        current_price = self.data[idx + self.seq_len - 1, 3]  # close
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(current_price, dtype=torch.float32),
        )


# =============================== loss ===============================
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
def denormalize_predictions(scaled_preds, scaler):
    """ Inverse standardization, restore original price scale (logarithm + inverse standardization operation)"""
    n, p = scaled_preds.shape
    original = np.zeros_like(scaled_preds)
    for i in range(p):
        dummy = np.zeros((n, 9))
        dummy[:, 3] = scaled_preds[:, i]  # close
        den = scaler.inverse_transform(dummy)
        original[:, i] = np.exp(den[:, 3]) - 1e-8
    return original


def preprocess_and_split_data(df: pd.DataFrame, look_back: int, pred_len: int):
    df_work = df.copy()
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

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
    #
    df_work['roll_return'] = (df_work['close'] - df_work['close'].shift(5)) / df_work['close'].shift(5)

    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)
    features = df_work[["open", "high", "low", "close", "volume", "open_interest", "ATR", "RSI", "roll_return"]].values
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


def train_model(model, train_loader, val_loader, epochs=160, lr=1e-4, model_dir='models_dlinear'):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8, verbose=False)
    crit = DirectionalLoss(alpha=1.0)
    best_val_loss = float('inf')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f"{model_dir}/dlinear_best.pth"
    patience = 0
    early_stop_patience = 8

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
        sch.step(val_loss)

        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # early stop
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
            preds_scaled.append(model(bx).cpu().numpy())
            trues_scaled.append(by.numpy())


    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues_scaled = np.concatenate(trues_scaled, axis=0)

    preds = denormalize_predictions(preds_scaled, scaler)
    trues = denormalize_predictions(trues_scaled, scaler)

    start_idx = train_size + 2 * look_back
    res_dates = dates_work[start_idx: start_idx + len(preds)]

    print("\n" + "=" * 80)
    print("（Step 1 ~ Step 12）")
    print("=" * 80)
    for step in range(PRED_LEN):
        mse_step = mean_squared_error(trues[:, step], preds[:, step])
        mae_step = mean_absolute_error(trues[:, step], preds[:, step])
        rmse_step = math.sqrt(mse_step)
        r2_step = r2_score(trues[:, step], preds[:, step])
        print(f"Step {step + 1:02d} - MSE: {mse_step:.6f}, MAE: {mae_step:.6f}, RMSE: {rmse_step:.6f}, R2: {r2_step:.6f}")

    # print
    print("\n" + "=" * 80)
    print("=" * 80)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(trues, preds)
    print(f"DLinear Model - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

    # DF
    df_out = pd.DataFrame()
    df_out['datetime'] = res_dates
    for k in range(preds.shape[1]):
        df_out[f'pred_step_{k + 1}'] = preds[:, k]
    df_out['pred_avg_1hr'] = preds.mean(axis=1)

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\nsave to: {save_path}")


# =============================== mian ===============================
def main(data_file="J_5.csv", output_dir="results_dlinear", model_dir="models_dlinear",
         pretrained_path: str | None = None, look_back_override: int | None = None):
    """
    Main function: Identical parameters and running logic to DLinearNHiTSAuto/BiLSTM
    Parameters:
    - data_file: Path to the input data file (CSV format)
    - output_dir: Directory for output results (for compatibility)
    - model_dir: Directory for saving the model
    - pretrained_path: Path to the pretrained model (optional)
    - look_back_override: Length of the input sequence (overrides the default value)
"""
    print(f"device: {device}")
    print(f"file: {data_file}")

    # 1. load data
    try:
        df = pd.read_csv(data_file)
        print(f"date loaded successfully， {len(df)} ")
    except FileNotFoundError:
        print(f"fail to find file '{data_file}'")
        return
    except Exception as e:
        print(f"fail：{e}")
        return

    # datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    if df["datetime"].isnull().any():
        df = df.dropna(subset=["datetime"])

    # 2. look_back
    lb_def = 48 if "30" in os.path.basename(data_file) else 20
    look_back = look_back_override if look_back_override is not None else lb_def
    print(f"length (look_back): {look_back}")
    print("begin...")
    try:
        train_loader, test_loader, scaler, dates_work, train_size, look_back = preprocess_and_split_data(
            df, look_back, PRED_LEN
        )
        print(f"Data preprocessing completed - Training set batches:{len(train_loader)}, Test set batches: {len(test_loader)}")
    except Exception as e:
        print(f"fail：{e}")
        return

    print("Initializing Standard DLinear Model...")
    model = DLinearModel(seq_len=look_back, pred_len=PRED_LEN, enc_in=6).to(device)
    print(f"Model initialization completed, total number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    if pretrained_path is not None and os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
            print(f"Successfully loaded pretrained weights: {pretrained_path}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")

    # 5. train
    print("begin DLinear...")
    try:
        model = train_model(model, train_loader, test_loader, epochs=160, lr=5e-3, model_dir=model_dir)
    except Exception as e:
        print(f"fail：{e}")
        return

    # 6. save
    print("begin...")
    save_path = "results/pred_dlinear.csv"
    try:
        generate_and_save_predictions(model, test_loader, scaler, dates_work, train_size, look_back, save_path)
    except Exception as e:
        print(f"fail：{e}")
        return

    print("\nfinished！")


# =============================== Enter ===============================
if __name__ == "__main__":


    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "results_dlinear"
        model_dir = sys.argv[3] if len(sys.argv) > 3 else "models_dlinear"
        pretrained = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "None" else None
        look_back_arg = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in ("", "None") else None
        main(data_file, output_dir, model_dir, pretrained_path=pretrained, look_back_override=look_back_arg)
    else:
        main()
"""
J

================================================================================
Step 01 - MSE: 64.767448, MAE: 5.457123, RMSE: 8.047823, R2: 0.994223
Step 02 - MSE: 90.032898, MAE: 6.403488, RMSE: 9.488567, R2: 0.991969
Step 03 - MSE: 113.402153, MAE: 7.192318, RMSE: 10.649045, R2: 0.989886
Step 04 - MSE: 140.726059, MAE: 8.107220, RMSE: 11.862801, R2: 0.987449
Step 05 - MSE: 167.549850, MAE: 8.937737, RMSE: 12.944105, R2: 0.985058
Step 06 - MSE: 191.633179, MAE: 9.564603, RMSE: 13.843164, R2: 0.982911
Step 07 - MSE: 217.781891, MAE: 10.196432, RMSE: 14.757435, R2: 0.980580
Step 08 - MSE: 248.120880, MAE: 10.991437, RMSE: 15.751853, R2: 0.977876
Step 09 - MSE: 272.717499, MAE: 11.455559, RMSE: 16.514161, R2: 0.975684
Step 10 - MSE: 297.435822, MAE: 12.064114, RMSE: 17.246328, R2: 0.973480
Step 11 - MSE: 331.512787, MAE: 13.100229, RMSE: 18.207493, R2: 0.970443
Step 12 - MSE: 349.127777, MAE: 13.249739, RMSE: 18.684961, R2: 0.968873


================================================================================
DLinear Model - MSE: 207.067261, MAE: 9.726665, RMSE: 14.389832, R2: 0.981536

"""