import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 1. Basic configuration
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
CONFIG = {
    "file_paths": {
        "base": r"pred_minute_advanced.csv",
        "patchtst": r"pred_patchtst.csv",
        "dlinear": r"pred_dlinear.csv",
        "itransformer": r"pred_itransformer.csv",
        "tsmixer": r"pred_tsmixer.csv",
        "true": r"I_true.csv"
    },
    "models": ["base", "patchtst", "dlinear", "itransformer", "tsmixer"],
    "n_steps": 12,  # 12-step forecast
    "true_col_prefix": "true_step_",
    "pred_col_prefix": "pred_step_",
    "alpha_threshold": 0.05,  # significance level
    "min_sample_size": 50,  # minimum sample size
    "loss_type": "MAE"  # loss type
}


# ==========================================
# Core function definitions
# ==========================================

def adf_stationarity_test(series, alpha=0.05):
    """ADF test to determine stationarity of the series"""
    try:
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value < alpha
        return {
            "p_value": p_value,
            "is_stationary": is_stationary
        }
    except:
        return {"p_value": 1.0, "is_stationary": False}


def auto_hac_lags(n, current_step=None):
    """
    Newey-West automatic HAC lag calculation.
    Fix: For h-step forecast, lag order should be at least h-1 to cover the moving average process.
    """
    # Basic Newey-West recommended formula
    nw_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    # If forecast horizon is specified, take the larger of the two
    if current_step is not None:
        min_lag = current_step - 1
        return max(nw_lag, min_lag)

    return nw_lag


def read_and_clean(file_path, is_true=False):
    """Read and clean data (【Fix】Remove interpolation to avoid future data leakage)"""
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # 1. Handle datetime column
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 2. Determine column names
    step_cols = [f"{CONFIG['true_col_prefix'] if is_true else CONFIG['pred_col_prefix']}{i}"
                 for i in range(1, CONFIG['n_steps'] + 1)]

    # 3. Convert to numeric
    for col in step_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. 【Key Fix】Directly drop rows containing missing values, strictly prohibit interpolation
    # This ensures that only real model-generated predictions participate in the test
    df = df.dropna(subset=step_cols)

    return df


def align_all_datasets(df_dict):
    """Align time indices of all models and true values"""
    common_dt = set(df_dict["true"]["datetime"])
    for model in CONFIG["models"]:
        common_dt.intersection_update(set(df_dict[model]["datetime"]))
    common_dt = sorted(list(common_dt))

    df_aligned = df_dict["true"][df_dict["true"]["datetime"].isin(common_dt)].copy()

    for model in CONFIG["models"]:
        df_model = df_dict[model][df_dict[model]["datetime"].isin(common_dt)].copy()
        pred_cols_old = [f"{CONFIG['pred_col_prefix']}{i}" for i in range(1, CONFIG['n_steps'] + 1)]
        pred_cols_new = [f"{model}_{CONFIG['pred_col_prefix']}{i}" for i in range(1, CONFIG['n_steps'] + 1)]
        df_model = df_model.rename(columns=dict(zip(pred_cols_old, pred_cols_new)))
        df_aligned = pd.merge(df_aligned, df_model[["datetime"] + pred_cols_new], on="datetime", how="inner")

    df_aligned = df_aligned.sort_values("datetime").reset_index(drop=True)
    print(f"Sample alignment completed, effective sample size: {len(df_aligned)}")
    if len(df_aligned) < CONFIG["min_sample_size"]:
        raise ValueError(f"Sample size too small ({len(df_aligned)}), unable to conduct valid test.")
    return df_aligned


def fdr_bh(p, q=0.05):
    """Benjamini-Hochberg FDR correction"""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0: return np.array([]), np.array([])

    valid_mask = ~np.isnan(p)
    p_valid = p[valid_mask]
    n_valid = p_valid.size

    if n_valid == 0: return np.full_like(p, np.nan), np.full_like(p, False)

    order = np.argsort(p_valid)
    p_sorted = p_valid[order]
    ranks = np.arange(1, n_valid + 1)
    p_adj_sorted = p_sorted * n_valid / ranks
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0, 1)

    p_adj = np.full_like(p, np.nan)
    p_adj[valid_mask] = p_adj_sorted[np.argsort(order)]
    sig = p_adj < q
    return p_adj, sig


def run_gw_test_final(loss_base, loss_comp, step_for_lag=None):
    """Perform GW test"""
    d_t = loss_comp - loss_base

    # Stationarity test
    adf_res = adf_stationarity_test(d_t)
    if not adf_res["is_stationary"]:
        return {"is_valid": False, "error": f"Non-stationary series (p={adf_res['p_value']:.4f})"}

    # Automatically calculate HAC lag order (pass current step to adjust lag)
    h = auto_hac_lags(len(d_t), current_step=step_for_lag)

    # OLS regression
    X = np.ones((len(d_t), 1))
    try:
        model = sm.OLS(d_t, X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    except Exception as e:
        return {"is_valid": False, "error": f"OLS fitting failed: {str(e)}"}

    gwstat = model.tvalues.iloc[0]
    const_coef = model.params.iloc[0]
    p_two_sided = model.pvalues.iloc[0]

    # One-sided p-value calculation (H0: Base == Comp, H1: Base < Comp i.e., Base is better)
    # If coef > 0, it means Comp has larger loss, Base is better
    if gwstat > 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - (p_two_sided / 2)

    dw_stat = durbin_watson(model.resid)
    is_better = (const_coef > 0) and (p_one_sided < CONFIG["alpha_threshold"])

    return {
        "is_valid": True,
        "gwstat": gwstat,
        "coef": const_coef,
        "p_one_sided": p_one_sided,
        "hac_lags": h,
        "dw_stat": dw_stat,
        "is_base_better_raw": is_better,
        "adf_p": adf_res["p_value"]
    }


def calculate_loss(df, model, step=None):
    if step is None:
        # 12-step average
        pred_cols = [f"{model}_{CONFIG['pred_col_prefix']}{i}" for i in range(1, CONFIG['n_steps'] + 1)]
        true_cols = [f"{CONFIG['true_col_prefix']}{i}" for i in range(1, CONFIG['n_steps'] + 1)]
        err = df[pred_cols].values - df[true_cols].values
    else:
        pred_col = f"{model}_{CONFIG['pred_col_prefix']}{step}"
        true_col = f"{CONFIG['true_col_prefix']}{step}"
        err = df[pred_col] - df[true_col]

    if CONFIG["loss_type"] == "MAE":
        loss = np.abs(err)
    elif CONFIG["loss_type"] == "MSE":
        loss = err ** 2

    return np.mean(loss, axis=1) if step is None else loss


# ==========================================
# 2. Main process execution
# ==========================================

# Read data
print("Reading data...")
df_dict = {}
df_dict["true"] = read_and_clean(CONFIG["file_paths"]["true"], is_true=True)
for model in CONFIG["models"]:
    df_dict[model] = read_and_clean(CONFIG["file_paths"][model], is_true=False)

# Align data
try:
    df_aligned = align_all_datasets(df_dict)
except ValueError as e:
    print(e)
    exit()

# Compute Base model loss
df_aligned["base_loss_avg"] = calculate_loss(df_aligned, "base", step=None)
for i in range(1, CONFIG['n_steps'] + 1):
    df_aligned[f"base_loss_step{i}"] = calculate_loss(df_aligned, "base", step=i)

# ------------------------------------------
# Stage 1: 12-step average loss test (displayed separately, not included in stepwise FDR correction)
# ------------------------------------------
print("\n" + "=" * 80)
print(f"[Stage 1] 12-step average loss test ({CONFIG['loss_type']})")
print("=" * 80)

avg_results = []
for model in CONFIG["models"]:
    if model == "base": continue

    loss_comp = calculate_loss(df_aligned, model, step=None)
    # The average loss series has strong serial correlation; use maximum step as reference lag
    res = run_gw_test_final(df_aligned["base_loss_avg"], loss_comp, step_for_lag=CONFIG["n_steps"])

    res["model"] = model
    avg_results.append(res)

    if res["is_valid"]:
        sig = "Significant" if res["is_base_better_raw"] else "Not significant"
        print(
            f"{model.upper()} vs BASE: GW={res['gwstat']:.4f}, p={res['p_one_sided']:.4e}, Lags={res['hac_lags']} -> {sig}")
    else:
        print(f"{model.upper()} vs BASE: Test failed ({res['error']})")

# ------------------------------------------
# Stage 2: Stepwise 1-12 loss tests (collect all p-values for global FDR correction)
# ------------------------------------------
print("\n" + "=" * 80)
print(f"[Stage 2] Stepwise tests ({CONFIG['n_steps']} steps) + global FDR correction")
print("=" * 80)

step_results_raw = []  # store all raw results
p_values_for_fdr = []  # store only valid p-values for correction
p_value_indices = []  # record corresponding indices (in step_results_raw) of p-values

current_idx = 0
for step in range(1, CONFIG['n_steps'] + 1):
    print(f"--- Step {step} ---")
    loss_base_step = df_aligned[f"base_loss_step{step}"]

    for model in CONFIG["models"]:
        if model == "base": continue

        loss_comp_step = calculate_loss(df_aligned, model, step=step)
        res = run_gw_test_final(loss_base_step, loss_comp_step, step_for_lag=step)

        res["model"] = model
        res["step"] = step
        step_results_raw.append(res)

        if res["is_valid"]:
            p_values_for_fdr.append(res["p_one_sided"])
            p_value_indices.append(current_idx)
            print(f"  {model}: p={res['p_one_sided']:.4e} (Lag={res['hac_lags']})")
        else:
            print(f"  {model}: Failed ({res.get('error')})")

        current_idx += 1

# ------------------------------------------
# Stage 3: Perform global FDR correction and fill back results
# ------------------------------------------
print("\nPerforming global FDR correction...")
if len(p_values_for_fdr) > 0:
    fdr_adj_p, fdr_sig = fdr_bh(p_values_for_fdr, q=CONFIG["alpha_threshold"])

    # Fill back into original results list
    for i, list_idx in enumerate(p_value_indices):
        step_results_raw[list_idx]["fdr_p"] = fdr_adj_p[i]
        step_results_raw[list_idx]["fdr_sig"] = fdr_sig[i]
else:
    print("No valid test results available for correction.")

# ------------------------------------------
# Stage 4: Generate final report
# ------------------------------------------
final_summary = []

# Add average loss results
for res in avg_results:
    row = {
        "Test Type": "12-step average",
        "Model": res["model"].upper(),
        "GW Statistic": res.get("gwstat", "-"),
        "HAC Lags": res.get("hac_lags", "-"),
        "One-sided p-value": res.get("p_one_sided", "-"),
        "FDR-adjusted p-value": "N/A (separate test)",
        "Raw Significant": res.get("is_base_better_raw", False),
        "FDR Significant": "N/A",
        "ADF p-value": res.get("adf_p", "-")
    }
    final_summary.append(row)

# Add stepwise results
for res in step_results_raw:
    row = {
        "Test Type": f"Step {res['step']}",
        "Model": res["model"].upper(),
        "GW Statistic": res.get("gwstat", "-"),
        "HAC Lags": res.get("hac_lags", "-"),
        "One-sided p-value": res.get("p_one_sided", "-"),
        "FDR-adjusted p-value": res.get("fdr_p", "-"),
        "Raw Significant": res.get("is_base_better_raw", False),
        "FDR Significant": res.get("fdr_sig", False),
        "ADF p-value": res.get("adf_p", "-")
    }
    final_summary.append(row)

# Save
summary_df = pd.DataFrame(final_summary)
output_file = "GW_test_results_fixed_version.csv"
summary_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print(f"All tests completed. Results saved to: {output_file}")
print("Guide to reading conclusions:")
print("1. [12-step average] Please refer to the 'Raw Significant' column, as it is a separate hypothesis.")
print("2. [Step 1-12] Please prioritize the 'FDR Significant' column, which controls false positive risk.")
print("=" * 80)

"""
================================================================================
[Stage 1] 12-step average loss test (MAE)
================================================================================
PATCHTST vs BASE: GW=13.9669, p=1.2408e-44, Lags=11 -> Significant
DLINEAR vs BASE: GW=12.3362, p=2.8927e-35, Lags=11 -> Significant
ITRANSFORMER vs BASE: GW=9.6727, p=1.9687e-22, Lags=11 -> Significant
TSMIXER vs BASE: GW=7.1073, p=5.9150e-13, Lags=11 -> Significant

================================================================================
[Stage 2] Stepwise tests (12 steps) + global FDR correction
================================================================================
--- Step 1 ---
  patchtst: p=2.2832e-97 (Lag=9)
  dlinear: p=3.1253e-81 (Lag=9)
  itransformer: p=7.1461e-56 (Lag=9)
  tsmixer: p=1.3097e-40 (Lag=9)
--- Step 2 ---
  patchtst: p=5.4597e-62 (Lag=9)
  dlinear: p=9.3772e-53 (Lag=9)
  itransformer: p=2.6550e-39 (Lag=9)
  tsmixer: p=5.7037e-20 (Lag=9)
--- Step 3 ---
  patchtst: p=1.9829e-46 (Lag=9)
  dlinear: p=4.2007e-38 (Lag=9)
  itransformer: p=1.4267e-31 (Lag=9)
  tsmixer: p=2.9866e-22 (Lag=9)
--- Step 4 ---
  patchtst: p=1.0500e-31 (Lag=9)
  dlinear: p=1.1506e-24 (Lag=9)
  itransformer: p=4.1602e-23 (Lag=9)
  tsmixer: p=1.8070e-14 (Lag=9)
--- Step 5 ---
  patchtst: p=1.1498e-29 (Lag=9)
  dlinear: p=2.4404e-30 (Lag=9)
  itransformer: p=2.1411e-20 (Lag=9)
  tsmixer: p=1.0804e-09 (Lag=9)
--- Step 6 ---
  patchtst: p=3.2814e-27 (Lag=9)
  dlinear: p=7.8674e-16 (Lag=9)
  itransformer: p=3.0807e-23 (Lag=9)
  tsmixer: p=3.5796e-07 (Lag=9)
--- Step 7 ---
  patchtst: p=1.1920e-25 (Lag=9)
  dlinear: p=6.6782e-15 (Lag=9)
  itransformer: p=1.1399e-18 (Lag=9)
  tsmixer: p=3.3222e-05 (Lag=9)
--- Step 8 ---
  patchtst: p=1.5655e-25 (Lag=9)
  dlinear: p=9.4918e-17 (Lag=9)
  itransformer: p=3.8293e-13 (Lag=9)
  tsmixer: p=4.6066e-06 (Lag=9)
--- Step 9 ---
  patchtst: p=7.8929e-20 (Lag=9)
  dlinear: p=1.0839e-12 (Lag=9)
  itransformer: p=2.8376e-10 (Lag=9)
  tsmixer: p=4.0781e-05 (Lag=9)
--- Step 10 ---
  patchtst: p=7.1650e-15 (Lag=9)
  dlinear: p=1.6617e-09 (Lag=9)
  itransformer: p=1.1186e-10 (Lag=9)
  tsmixer: p=2.9002e-03 (Lag=9)
--- Step 11 ---
  patchtst: p=4.9798e-12 (Lag=10)
  dlinear: p=6.9860e-10 (Lag=10)
  itransformer: p=1.7879e-07 (Lag=10)
  tsmixer: p=7.3169e-03 (Lag=10)
--- Step 12 ---
  patchtst: p=1.1122e-10 (Lag=11)
  dlinear: p=8.9500e-10 (Lag=11)
  itransformer: p=1.8678e-07 (Lag=11)
  tsmixer: p=8.2523e-03 (Lag=11)
"""