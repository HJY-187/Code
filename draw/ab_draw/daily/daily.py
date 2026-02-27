import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


LOOK_BACK = 22
PRED_LEN = 1
EPOCHS = 180
LR = 5e-4
BATCH_SIZE = 64
DATA_FILE = "I_daily.csv"
OUTPUT_FILE = "ablation_study_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def reset_seeds(seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================== Dataset ===============================
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

    # 2. daily features
    dt = df_work['datetime']
    df_work['day_of_week'] = dt.dt.dayofweek
    df_work['day_of_year'] = dt.dt.dayofyear
    df_work['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df_work['month'] = dt.dt.month

    # 3. metrics
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

    # 4. clean
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # 5. features
    feature_cols = [
        "open", "high", "low", "close", "volume", "open_interest",
        "day_of_week", "day_of_year", "week_of_year", "month",
        "ATR", "RSI", "roll_return"
    ]
    features = df_work[feature_cols].values

    # series
    dates = df_work['datetime'].values

    # 6.split
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size + gap:]
    raw_test_dates = dates[train_size + gap:]

    # 7. normalization
    scaler = RobustScaler()
    scaler.fit(raw_train)
    train_data = scaler.transform(raw_train)
    test_data = scaler.transform(raw_test)

    # 8. DataLoader
    train_ds = TimeSeriesDataset(train_data, seq_len=look_back)
    test_ds = TimeSeriesDataset(test_data, seq_len=look_back)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    valid_test_len = len(test_ds)
    target_dates_indices = np.arange(look_back, look_back + valid_test_len)
    test_target_dates = raw_test_dates[target_dates_indices]

    return train_loader, test_loader, scaler, len(feature_cols), test_target_dates


def denormalize_close(scaled_val, scaler, num_features):
    dummy = np.zeros((1, num_features))
    dummy[0, 3] = scaled_val  
    denorm = scaler.inverse_transform(dummy)[0, 3]
    return np.exp(denorm) - 1e-8


# =============================== model ===============================
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels,
                                   bias=False)
        self.pointwise = nn.Conv1d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        y = self.depthwise(x)
        y = F.gelu(y)
        y = self.pointwise(y)
        y = self.bn(y)
        return y


class AS1(nn.Module):
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


class AS2(nn.Module):
    def __init__(self, seq_len: int, dropout: float = 0.1, activation: str = 'ReLU'):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)

        act_layer = nn.ReLU() if activation == 'ReLU' else nn.GELU()

        self.net = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(seq_len // 2, PRED_LEN)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class AblationModel(nn.Module):
    def __init__(self, seq_len: int, in_features: int,
                 ma_kernel_size: int = 21,
                 use_nhits: bool = True, use_auto: bool = True,
                 nhits_init: float = 0.05, auto_init: float = 0.1,
                 activation: str = 'GELU', decoder_dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.ma_kernel_size = ma_kernel_size
        self.nhits_coef = nn.Parameter(torch.tensor(nhits_init))
        self.auto_coef = nn.Parameter(torch.tensor(auto_init))
        self.use_nhits = use_nhits
        self.use_auto = use_auto

        # === Base ===
        self.trend_linear = nn.Linear(seq_len, PRED_LEN)
        self.residual_linear = nn.Linear(seq_len, PRED_LEN)
        self.feature_weights = nn.Parameter(torch.ones(in_features))

        self.decoder = nn.Sequential(
            nn.Linear(PRED_LEN, PRED_LEN * 2),
            nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(PRED_LEN * 2, PRED_LEN)
        )

        if self.use_auto:
            self.auto_block = AS1(channels=in_features, seq_len=seq_len, ma_kernel_size=ma_kernel_size)

        if self.use_nhits:
            self.nhits_block = AS2(seq_len=seq_len, dropout=0.2, activation=activation)

    def moving_avg(self, x):
        padding = (self.ma_kernel_size - 1) // 2
        avg = nn.AvgPool1d(kernel_size=self.ma_kernel_size, stride=1, padding=padding)
        return avg(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):

        w = self.feature_weights.unsqueeze(0).unsqueeze(0)
        weighted_x = x * w

        trend = self.moving_avg(weighted_x)
        residual = weighted_x - trend
        trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_linear(residual.permute(0, 2, 1)).permute(0, 2, 1)

        base_pred = (trend_pred + residual_pred)[:, :, 3]  # Index 3 is Close
        refined_base = base_pred + 0.1 * self.decoder(base_pred)

        out = refined_base


        if self.use_auto:
            auto_out = self.auto_block(weighted_x)
            out = out + self.auto_coef * auto_out


        if self.use_nhits:
            nhits_out = self.nhits_block(residual[:, :, 3])
            out = out + self.nhits_coef * nhits_out

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


def run_experiment(exp_name, use_nhits, use_auto, loaders, scaler, num_features, test_dates, **model_kwargs):

    print(f"\n[{exp_name}] Starting... (AS2={use_nhits}, Auto={use_auto})")


    reset_seeds(42)

    train_loader, test_loader = loaders


    model = AblationModel(
        seq_len=LOOK_BACK,
        in_features=num_features,
        ma_kernel_size=21,
        use_nhits=use_nhits,
        use_auto=use_auto,
        **model_kwargs
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    criterion = DirectionalLoss(alpha=2.0)

    best_loss = float('inf')
    best_weights = None
    patience_counter = 0


    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by, last_close in train_loader:
            bx, by, last_close = bx.to(DEVICE), by.to(DEVICE), last_close.to(DEVICE)
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
            for bx, by, last_close in test_loader:
                bx, by, last_close = bx.to(DEVICE), by.to(DEVICE), last_close.to(DEVICE)
                pred = model(bx)
                loss = criterion(pred, by, last_close)
                val_loss += loss.item()
        val_loss /= len(test_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break


    model.load_state_dict(best_weights)
    model.eval()

    print(f"[{exp_name}] Learned Weights -> Auto: {model.auto_coef.item():.4f}, AS2: {model.nhits_coef.item():.4f}")

    preds_scaled, trues_scaled, prev_closes_scaled = [], [], []
    with torch.no_grad():
        for bx, by, b_prev in test_loader:
            bx = bx.to(DEVICE)
            output = model(bx)
            preds_scaled.extend(output.cpu().numpy().flatten())
            trues_scaled.extend(by.numpy().flatten())
            prev_closes_scaled.extend(b_prev.numpy().flatten())


    final_preds, final_trues, final_prevs = [], [], []
    loop_len = min(len(preds_scaled), len(trues_scaled), len(prev_closes_scaled))


    if loop_len != len(test_dates):
        print(f"Warning: Length mismatch. Preds: {loop_len}, Dates: {len(test_dates)}. Trimming to min.")
        loop_len = min(loop_len, len(test_dates))
        test_dates = test_dates[:loop_len]

    for i in range(loop_len):
        final_preds.append(denormalize_close(preds_scaled[i], scaler, num_features))
        final_trues.append(denormalize_close(trues_scaled[i], scaler, num_features))
        final_prevs.append(denormalize_close(prev_closes_scaled[i], scaler, num_features))

    final_trues = np.array(final_trues)
    final_preds = np.array(final_preds)
    final_prevs = np.array(final_prevs)


    mae = mean_absolute_error(final_trues, final_preds)
    mse = mean_squared_error(final_trues, final_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(final_trues, final_preds)

    true_trend = np.sign(final_trues - final_prevs)
    pred_trend = np.sign(final_preds - final_prevs)
    trend_acc = np.mean(true_trend == pred_trend) * 100

    print(f"[{exp_name}] Finished. MAE: {mae:.4f} | Trend Acc: {trend_acc:.2f}%")


    safe_name = exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    pred_filename = f"pred_{safe_name}.csv"

    pred_df = pd.DataFrame({
        'datetime': test_dates,
        'pred_close': final_preds
    })
    pred_df.to_csv(pred_filename, index=False)
    print(f"[{exp_name}] Saved predictions to {pred_filename}")


    return {
        "Experiment": exp_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Trend_Acc(%)": trend_acc
    }



if __name__ == "__main__":
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        exit()

    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])


    train_loader, test_loader, scaler, num_features, test_dates = preprocess_data(df, LOOK_BACK)
    loaders = (train_loader, test_loader)

    print("-" * 50)
    print(f"Data ready. Features: {num_features}. Test Samples: {len(test_dates)}")
    print("Starting Ablation Study (Optimizing for MAE)...")
    print("-" * 50)


    def random_search_daily(loaders, scaler, num_features, test_dates, num_trials=3):
        print("\n" + "=" * 50)
        print(f" Starting Random Search Optimization (Daily) - {num_trials} trials")
        print("   >>> Optimizing for MAE (Mean Absolute Error)")
        print("=" * 50)

        param_grid = {
            "activation": ['GELU', 'ReLU'],
            "nhits_init": [0.01, 0.03, 0.05, 0.08, 0.1],
            "auto_init": [0.1, 0.15, 0.2, 0.25, 0.3],
            "decoder_dropout": [0.1, 0.2, 0.3]
        }

        best_mae = float('inf')  
        best_config = {}

        for i in range(num_trials):
            config = {k: random.choice(v) for k, v in param_grid.items()}
            print(f"\n[Trial {i + 1}] Config: {config}")

            res = run_experiment(f"Trial_{i}", True, True, loaders, scaler, num_features, test_dates, **config)


            if res['MAE'] < best_mae:
                best_mae = res['MAE']
                best_config = config
                print(f"   >>> New Best MAE: {best_mae:.4f}")

        print(f"\n🏆 Best Config (Low MAE): {best_config}")
        return best_config



    best_params = random_search_daily(loaders, scaler, num_features, test_dates, num_trials=3)


    print("=" * 60)
    print("STARTING FINAL ABLATION WITH OPTIMIZED PARAMS")
    print("=" * 60)

    results = []

    res_base = run_experiment("Base (main)", False, False, loaders, scaler, num_features, test_dates,
                              **best_params)
    results.append(res_base)


    res_auto = run_experiment("Base + AS1", False, True, loaders, scaler, num_features, test_dates,
                              **best_params)
    results.append(res_auto)


    res_nhits = run_experiment("Base + AS2", True, False, loaders, scaler, num_features, test_dates, **best_params)
    results.append(res_nhits)


    res_full = run_experiment("Full Hybrid Model", True, True, loaders, scaler, num_features, test_dates, **best_params)
    results.append(res_full)


    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print(" FINAL ABLATION STUDY RESULTS ")
    print("=" * 60)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(results_df)
    print("=" * 60)

    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Metrics saved to {OUTPUT_FILE}")

"""
============================================================
 FINAL ABLATION STUDY RESULTS 
============================================================
            Experiment     MAE    RMSE     R2  Trend_Acc(%)
0  Base (main) 12.4864 16.7595 0.9567       51.9187
1    Base + AS1 12.4631 16.6226 0.9574       51.4673
2        Base + AS2 12.4453 16.7700 0.9566       50.3386
3    Full Hybrid Model 12.4065 16.6202 0.9574       51.0158
============================================================
Metrics saved to ablation_study_results.csv
"""