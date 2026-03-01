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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Added: Import matplotlib font configuration module
import matplotlib as mpl

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================== Configuration Parameters ===============================
# Appropriately increase the lookback window to capture longer-cycle calendar features
LOOK_BACK = 30
PRED_LEN = 1
EPOCHS = 180
LR = 5e-4
BATCH_SIZE = 64
DATA_FILE = "J_daily.csv"
# Fix 1: Specify output path with directory (prioritize output folder in current directory to avoid empty directory issues)
OUTPUT_DIR = "output"  # Added: Define output directory
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pred_daily_advanced_J.csv")  # Modified: Concatenate directory and filename
IMG_FILE = os.path.join(OUTPUT_DIR, "pred_daily_chart_J.png")  # Modified: Also put image in output directory


# =============================== Dataset ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - PRED_LEN + 1

    def __getitem__(self, idx: int):
        # Input: All features from the previous seq_len rows
        x = self.data[idx: idx + self.seq_len, :]
        # Target: close value at seq_len position (index 3)
        y = self.data[idx + self.seq_len, 3]
        # Current moment's closing price (used for calculating trend change)
        previous_close = self.data[idx + self.seq_len - 1, 3]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
            torch.tensor(previous_close, dtype=torch.float32)
        )


# =============================== Data Preprocessing ===============================
def preprocess_data(df: pd.DataFrame, look_back: int):
    df_work = df.copy()

    # 1. Log transformation
    for col in ["open", "high", "low", "close"]:
        df_work[col] = np.log(df_work[col] + 1e-8)

    # 2. Add calendar features (Calendar Features)
    dt = df_work['datetime']
    df_work['day_of_week'] = dt.dt.dayofweek
    df_work['day_of_year'] = dt.dt.dayofyear
    df_work['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df_work['month'] = dt.dt.month

    # 3. Technical indicator calculation
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

    # 4. Clean data
    df_work = df_work.iloc[14:].dropna().reset_index(drop=True)

    # 5. Extract features
    feature_cols = [
        "open", "high", "low", "close", "volume", "open_interest",
        "day_of_week", "day_of_year", "week_of_year", "month",
        "ATR", "RSI", "roll_return"
    ]
    features = df_work[feature_cols].values

    # 6. Split train/test sets
    total = len(features)
    train_size = int(total * 0.8)
    gap = look_back

    raw_train = features[:train_size]
    raw_test = features[train_size:]  # No gap added here to ensure test set start connects properly

    # Extract target timestamps corresponding to test set
    # In raw_test length, the first look_back entries are used to predict the (look_back+1)th data point
    # So the index corresponding to the first predicted point is actually look_back
    test_target_dates = df_work['datetime'].iloc[train_size + look_back:].values

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

    return train_loader, test_loader, scaler, test_target_dates, len(feature_cols)


# =============================== Model Components ===============================
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
    def __init__(self, seq_len: int, in_features: int, ma_kernel_size: int = 21,
                 nhits_init: float = 0.01, auto_init: float = 0.2, activation: str = 'GELU'):
        super().__init__()
        self.in_features = in_features
        self.ma_kernel_size = ma_kernel_size
        self.nhits_coef = nn.Parameter(torch.tensor(nhits_init))
        self.auto_coef = nn.Parameter(torch.tensor(auto_init))

        self.trend_linear = nn.Linear(seq_len, PRED_LEN)
        self.residual_linear = nn.Linear(seq_len, PRED_LEN)
        self.feature_weights = nn.Parameter(torch.ones(in_features))

        self.decoder = nn.Sequential(
            nn.Linear(PRED_LEN, PRED_LEN * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(PRED_LEN * 2, PRED_LEN)
        )
        self.auto_block = AS1(channels=in_features, seq_len=seq_len, ma_kernel_size=ma_kernel_size)
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

        # Base DLinear part only takes Close channel (index 3)
        base_pred = (trend_pred + residual_pred)[:, :, 3]
        refined_base = base_pred + 0.1 * self.decoder(base_pred)

        # Auxiliary modules
        auto_out = self.auto_block(weighted_x)
        nhits_out = self.nhits_block(residual[:, :, 3])

        out = refined_base + self.auto_coef * auto_out + self.nhits_coef * nhits_out
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
        # Penalty only when directions are opposite (product is negative, negated becomes positive)
        penalty = F.relu(-1.0 * true_diff * pred_diff)
        return num_loss + self.alpha * torch.mean(penalty)


# =============================== Utility Functions ===============================
def denormalize_close(scaled_val_list, scaler, num_features):
    """Batch denormalization"""
    dummy = np.zeros((len(scaled_val_list), num_features))
    dummy[:, 3] = scaled_val_list
    denorm = scaler.inverse_transform(dummy)[:, 3]
    return np.exp(denorm) - 1e-8


def run_process():
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 1. Prepare data
    train_loader, test_loader, scaler, test_target_dates, num_features = preprocess_data(df, LOOK_BACK)
    print(f"Features: {num_features}, Test samples: {len(test_target_dates)}")

    # 2. Initialize model
    model = DENet(
        seq_len=LOOK_BACK,
        in_features=num_features,
        ma_kernel_size=21,
        nhits_init=0.01,
        auto_init=0.2,
        activation='GELU'
    ).to(device)

    # 3. Training configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = DirectionalLoss(alpha=2.0)

    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    print("Starting training...")
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
            for bx, by, last_close in test_loader:
                bx, by, last_close = bx.to(device), by.to(device), last_close.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by, last_close).item()
        val_loss /= len(test_loader)

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 4. Load best model for evaluation
    print("Evaluating best model...")
    model.load_state_dict(best_model_weights)
    model.eval()

    preds_scaled = []
    trues_scaled = []
    prev_closes_scaled = []

    with torch.no_grad():
        for bx, by, b_prev_c in test_loader:
            bx = bx.to(device)
            output = model(bx)
            preds_scaled.extend(output.cpu().numpy().flatten())
            trues_scaled.extend(by.numpy().flatten())
            prev_closes_scaled.extend(b_prev_c.numpy().flatten())

    # 5. Denormalization
    final_preds = denormalize_close(preds_scaled, scaler, num_features)
    final_trues = denormalize_close(trues_scaled, scaler, num_features)
    final_prev_closes = denormalize_close(prev_closes_scaled, scaler, num_features)

    # 6. Calculate metrics
    mae = mean_absolute_error(final_trues, final_preds)
    mse = mean_squared_error(final_trues, final_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(final_trues, final_preds)

    # Calculate Trend Accuracy (direction accuracy)
    true_diff = final_trues - final_prev_closes
    pred_diff = final_preds - final_prev_closes

    # Filter out points with extremely small changes to avoid noise interference
    mask = np.abs(true_diff) > 1e-6
    if np.sum(mask) > 0:
        correct_direction = np.sign(true_diff[mask]) == np.sign(pred_diff[mask])
        trend_acc = np.mean(correct_direction) * 100
    else:
        trend_acc = 0.0

    print("\n" + "=" * 40)
    print(" Evaluation Metrics (Test Set) ")
    print("=" * 40)
    print(f" MAE          : {mae:.4f}")
    print(f" RMSE         : {rmse:.4f}")
    print(f" R^2          : {r2:.4f}")
    print(f" Trend Acc    : {trend_acc:.2f}%")
    print("=" * 40 + "\n")

    # 7. Save CSV
    # Fix 2: Create output directory first (now directory is not empty, won't cause errors)
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Modified: Directly create the defined OUTPUT_DIR

    min_len = min(len(test_target_dates), len(final_preds))
    results_df = pd.DataFrame({
        "datetime": test_target_dates[:min_len],
        "true_close": final_trues[:min_len],
        "pred_close": final_preds[:min_len]
    })

    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to: {OUTPUT_FILE}")

    # 8. Plot and save image (modified version)
    # ========== Key Modification 1: Set global font to Times New Roman ==========
    # Compatible font names for different systems (Windows: Times New Roman; Mac/Linux: TimesRoman)
    try:
        # Windows system
        mpl.rcParams['font.family'] = 'Times New Roman'
    except:
        # Mac/Linux system fallback
        mpl.rcParams['font.family'] = 'TimesRoman'
    # Set unified font size baseline
    mpl.rcParams['font.size'] = 12
    # Disable abnormal minus sign display issue
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(14, 7))

    # Convert datetime to plotting format
    dates = results_df['datetime']

    # Modified to example chart colors: Ground Truth in blue, Prediction in orange
    plt.plot(dates, results_df['true_close'], label='Ground Truth', color='#1f77b4', alpha=0.8, linewidth=1.5)
    plt.plot(dates, results_df['pred_close'], label='Prediction', color='#ff7f0e', alpha=0.9, linewidth=1)

    # ========== Key Modification 2: Explicitly specify title/label font as Times New Roman ==========
    plt.title("prediction vs true + DENet(J)", fontsize=18, fontweight='bold', fontfamily='Times New Roman')
    plt.xlabel("Time Step", fontsize=12, fontfamily='Times New Roman')
    plt.ylabel("Price", fontsize=12, fontfamily='Times New Roman')
    plt.legend(loc="upper right", fontsize=12, prop={'family': 'Times New Roman'})
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set X-axis date format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()

    # Save image
    plt.savefig(IMG_FILE, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {IMG_FILE}")


if __name__ == "__main__":
    run_process()