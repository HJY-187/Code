import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
import sys

warnings.filterwarnings("ignore")

# ----------------------------
# Configuration Parameters
# ----------------------------
CONFIG = {
    "file_paths": {
        "true": "J_true.csv",
        "base": "pred_daily_advanced.csv",
        "patchtst": "pred_daily_patchtst.csv",
        "dlinear": "pred_daily_simple_dlinear.csv",
        "itransformer": "pred_daily_itransformer.csv",
        "tsmixer": "pred_daily_tsmixer_revin.csv",
    },
    "models": ["base", "patchtst", "dlinear", "itransformer", "tsmixer"],
    "base_model": "base",

    # Significance level
    "alpha_threshold": 0.05,

    # HAC lags
    "hac_lags_default": None,  # None => automatic

    # Minimum sample size
    "min_sample_size": 30,

    # Loss function
    "loss_type": "MSE",  # Currently only MSE implemented

    # File encoding attempts
    "encoding_candidates": ["utf-8-sig", "utf-8", "gbk", "gb2312"],

    # Prediction misalignment check
    # Alignment candidates: 0 = same-day alignment; +1 = shift prediction forward by one day
    "alignment_shift_candidates": [0, 1],

    # Number of sample days printed (manual verification)
    "sample_print_days": 3,

    # Threshold for automatic shift preference strength
    # Rule: if best_shift MSE < second_best * (1 - improvement_ratio), strongly suggest misalignment
    "shift_improvement_ratio": 0.02,  # 2% improvement triggers hint
}

# ----------------------------
# Plot font settings
# ----------------------------
plt.rcParams["axes.unicode_minus"] = False
try:
    import platform

    sys_str = platform.system()
    if sys_str == "Windows":
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    elif sys_str == "Darwin":
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC"]
    else:
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
except Exception:
    pass


# ----------------------------
# Utility Functions
# ----------------------------
def pick_first_existing(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Column not found, candidates={candidates}")
    return None


def read_csv_with_encoding(path: str) -> pd.DataFrame:
    last_err = None
    for encoding in CONFIG["encoding_candidates"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Unable to read file: {path}, last error: {last_err}")


def preprocess_df(df: pd.DataFrame, val_col_name: str, model_tag: str) -> pd.DataFrame:
    """Standardize datetime column and value column"""
    dt_col = pick_first_existing(df, ["datetime", "date", "dt", "time"], required=False)

    if not dt_col:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            dt_col = df.columns[0]
        else:
            raise ValueError(f"No datetime column found in model {model_tag} data")

    target_candidates = [
        f"{model_tag}_pred", f"{model_tag}", val_col_name,
        "pred_close", "pred", "close", "true_close"
    ]
    val_col = pick_first_existing(df, target_candidates)

    out = df.rename(columns={dt_col: "datetime", val_col: val_col_name}).copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out[val_col_name] = pd.to_numeric(out[val_col_name], errors="coerce")
    out = out.dropna(subset=["datetime", val_col_name])[["datetime", val_col_name]].sort_values("datetime")

    # Deduplicate: if multiple rows per day, keep the last one (can change to mean if needed)
    out = out.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    return out


def load_all_data() -> Dict[str, pd.DataFrame]:
    data_dict: Dict[str, pd.DataFrame] = {}
    print("Reading true values...")
    df_true = read_csv_with_encoding(CONFIG["file_paths"]["true"])
    data_dict["true"] = preprocess_df(df_true, "true_close", "true")

    for m in CONFIG["models"]:
        print(f"Reading model: {m}...")
        try:
            df_m = read_csv_with_encoding(CONFIG["file_paths"][m])
            data_dict[m] = preprocess_df(df_m, f"{m}_pred", m)
        except Exception as e:
            print(f"Warning: Failed to read model {m}: {e}")

    return data_dict


# ----------------------------
# Loss & Alignment
# ----------------------------
def compute_loss(true: pd.Series, pred: pd.Series, loss_type: str = "MSE") -> pd.Series:
    if loss_type.upper() == "MSE":
        return (pred - true) ** 2
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def merge_for_models(data: Dict[str, pd.DataFrame], base_model: str, comp_model: str, shift: int = 0) -> pd.DataFrame:
    """
    shift meaning:
    - shift=0: align prediction with same-day true
    - shift=1: shift prediction forward by one day (pred(t) -> used for true(t+1))
      (common case: predicting next day but saved with date t)
    """
    df_true = data["true"].copy()
    df_base = data[base_model].copy()
    df_comp = data[comp_model].copy()

    if shift != 0:
        # Move datetime instead of shifting rows (more robust for missing trading days)
        df_base["datetime"] = df_base["datetime"] + pd.to_timedelta(shift, unit="D")
        df_comp["datetime"] = df_comp["datetime"] + pd.to_timedelta(shift, unit="D")

    df = pd.merge(df_true, df_base, on="datetime", how="inner")
    df = pd.merge(df, df_comp, on="datetime", how="inner")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def alignment_check_and_hint(data: Dict[str, pd.DataFrame], base_model: str, comp_models: List[str]) -> int:
    """
    Automatic alignment check: compare MSE under different shifts (based on base model), choose optimal shift.
    Also prints sample days for manual verification.
    Returns suggested shift (not forcibly applied; main program will use suggestion).
    """
    shifts = CONFIG["alignment_shift_candidates"]
    loss_type = CONFIG["loss_type"]

    if base_model not in data or "true" not in data:
        print("Alignment check unavailable: missing true or base data. Default shift=0.")
        return 0

    shift_mse: List[Tuple[int, float, int]] = []
    for s in shifts:
        df_true = data["true"].copy()
        df_base = data[base_model].copy()
        if s != 0:
            df_base["datetime"] = df_base["datetime"] + pd.to_timedelta(s, unit="D")
        df0 = pd.merge(df_true, df_base, on="datetime", how="inner").sort_values("datetime")
        if len(df0) < CONFIG["min_sample_size"]:
            continue
        loss = compute_loss(df0["true_close"], df0[f"{base_model}_pred"], loss_type=loss_type)
        shift_mse.append((s, float(loss.mean()), int(len(df0))))

    if not shift_mse:
        print("Insufficient merged samples for alignment check. Default shift=0.")
        return 0

    shift_mse = sorted(shift_mse, key=lambda x: x[1])
    best_shift, best_mse, best_n = shift_mse[0]
    print("\n" + "=" * 80)
    print("Automatic prediction alignment check (based on Base vs True MSE)")
    for s, mse, n in shift_mse:
        print(f"  shift={s:+d} day(s) | MSE={mse:.6f} | n={n}")
    print(f"Suggested alignment: shift={best_shift:+d} (lowest MSE)")

    if len(shift_mse) >= 2:
        _, second_mse, _ = shift_mse[1]
        improvement = (second_mse - best_mse) / max(second_mse, 1e-12)
        if improvement >= CONFIG["shift_improvement_ratio"]:
            print(f"Hint: shift={best_shift:+d} improves MSE by about {improvement * 100:.2f}% over second best")
            print("   This usually indicates potential misalignment between saved date and target prediction date.")
            print("   Please manually verify using the sample below.")
        else:
            print("Shift advantage is small; misalignment signal is weak, but manual verification is still recommended.")

    comp_for_print = None
    for m in comp_models:
        if m in data:
            comp_for_print = m
            break

    try:
        if comp_for_print:
            df_print = merge_for_models(data, base_model, comp_for_print, shift=best_shift)
            print("\nSample verification (first few aligned days):")
            cols = ["datetime", "true_close", f"{base_model}_pred", f"{comp_for_print}_pred"]
            print(df_print[cols].head(CONFIG["sample_print_days"]).to_string(index=False))
            print("   Note: Check whether pred appears to predict the corresponding date's true value.")
        else:
            df_true = data["true"].copy()
            df_base = data[base_model].copy()
            if best_shift != 0:
                df_base["datetime"] = df_base["datetime"] + pd.to_timedelta(best_shift, unit="D")
            df_print = pd.merge(df_true, df_base, on="datetime", how="inner").sort_values("datetime")
            print("\nSample verification (first few aligned days):")
            cols = ["datetime", "true_close", f"{base_model}_pred"]
            print(df_print[cols].head(CONFIG["sample_print_days"]).to_string(index=False))
    except Exception as e:
        print(f"Sample printing failed: {e}")

    print("=" * 80 + "\n")
    return best_shift


# ----------------------------
# Core Statistical Test (HAC mean test, DM/GW style)
# ----------------------------
def hac_test_pairwise(loss_diff: pd.Series, lags: Optional[int] = None) -> Optional[dict]:
    """
    Perform: OLS(loss_diff ~ 1) + HAC(Newey-West) covariance
    GW_stat = t-statistic (commonly used as GW/DM-style statistic)
    One-sided test direction:
      H1: E[diff] > 0  => comp worse / base better
    """
    n = int(loss_diff.dropna().shape[0])
    if n < CONFIG["min_sample_size"]:
        return None

    d = loss_diff.dropna().values
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
    lags = max(1, int(lags))

    X = np.ones((n, 1))
    try:
        model = sm.OLS(d, X)
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})

        alpha = float(results.params[0])
        gw_stat = float(results.tvalues[0])
        p_two = float(results.pvalues[0])

        if gw_stat >= 0:
            p_one = p_two / 2.0
        else:
            p_one = 1.0 - p_two / 2.0

        return {
            "alpha": alpha,
            "GW_stat": gw_stat,
            "p_two": p_two,
            "p_one": p_one,
            "lags": lags,
            "n": n,
        }
    except Exception as e:
        print(f"OLS/HAC computation error: {e}")
        return None


def fdr_bh(p_values: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    i = np.argsort(p)
    m = np.argsort(i)
    p_sorted = p[i]
    rank = np.arange(1, n + 1)
    bh_val = p_sorted * n / rank
    p_adj_sorted = np.minimum.accumulate(bh_val[::-1])[::-1]
    p_adj_sorted = np.minimum(p_adj_sorted, 1.0)
    p_adj = p_adj_sorted[m]
    return p_adj, p_adj < alpha


# ----------------------------
# Plotting
# ----------------------------
def plot_results(df: pd.DataFrame, base_model: str, out_img: str = "daily_significance_gw_plot.png") -> None:
    plt.figure(figsize=(12, 6), dpi=150)

    colors = ["#d62728" if bool(row["is_significant"]) else "#1f77b4" for _, row in df.iterrows()]
    bars = plt.bar(df["model"], df["comp_loss"], color=colors, alpha=0.8, edgecolor="k")

    plt.axhline(df["base_loss"].mean(), color="black", linestyle="--", label=f"Base ({base_model}) Avg Loss")

    for bar, sig, p_val, gw in zip(bars, df["is_significant"], df["p_fdr"], df["GW_stat"]):
        height = bar.get_height()
        label = f"GW={gw:.2f}\n" + ("*" if sig else "") + f"(p={p_val:.3e})"
        font_weight = "bold" if sig else "normal"
        color = "red" if sig else "black"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.01,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            fontweight=font_weight,
        )

    plt.title("Model Prediction Loss Comparison (HAC Mean Test / DM-GW Style + FDR Correction)", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.savefig(out_img, bbox_inches="tight")
    print(f"Image saved to: {out_img} (please open the file directly)")
    # Do not call plt.show() to avoid backend issues in some IDEs


# ----------------------------
# Main Program
# ----------------------------
def main():
    data = load_all_data()
    base_model = CONFIG["base_model"]

    if base_model not in data or "true" not in data:
        raise ValueError("Missing true values or Base model data")

    comp_models = [m for m in CONFIG["models"] if m != base_model and m in data]
    if not comp_models:
        raise ValueError("No available comparison model data (comp_models is empty)")

    suggested_shift = alignment_check_and_hint(data, base_model, comp_models)

    print("\n" + "=" * 80)
    print("Starting pairwise significance test (HAC mean test / DM-GW style)")
    print("Test direction: diff = loss_comp - loss_base")
    print("   H1: E[diff] > 0  => comp worse / base better (t>0 supports base better)")
    print(f"Current alignment used: shift={suggested_shift:+d} day(s)")
    print("   Practical tip: GW_stat > 1.645 usually indicates one-sided 0.05 significance (large sample approx)")
    print("=" * 80)

    results_list = []
    loss_type = CONFIG["loss_type"]

    for m in comp_models:
        df_merged = merge_for_models(data, base_model, m, shift=suggested_shift)
        if len(df_merged) < CONFIG["min_sample_size"]:
            print(f"{m}: insufficient merged samples (n={len(df_merged)}), skipped")
            continue

        loss_base = compute_loss(df_merged["true_close"], df_merged[f"{base_model}_pred"], loss_type=loss_type)
        loss_comp = compute_loss(df_merged["true_close"], df_merged[f"{m}_pred"], loss_type=loss_type)

        loss_diff = (loss_comp - loss_base)

        res = hac_test_pairwise(loss_diff, lags=CONFIG["hac_lags_default"])
        if not res:
            print(f"{m}: HAC test failed or insufficient samples")
            continue

        direction = "base_better" if res["GW_stat"] > 0 else "comp_better"

        results_list.append({
            "model": m,
            "n_samples": res["n"],
            "alignment_shift_days": suggested_shift,
            "base_loss": float(loss_base.mean()),
            "comp_loss": float(loss_comp.mean()),
            "diff_mean": res["alpha"],
            "GW_stat": res["GW_stat"],
            "p_one(H1:base_better)": res["p_one"],
            "p_two": res["p_two"],
            "lags": res["lags"],
            "direction_by_t": direction,
        })

    res_df = pd.DataFrame(results_list)
    if res_df.empty:
        print("No valid test results.")
        return

    res_df["p_fdr"], res_df["is_significant"] = fdr_bh(
        res_df["p_one(H1:base_better)"].tolist(),
        alpha=CONFIG["alpha_threshold"]
    )
    res_df = res_df.sort_values("comp_loss").reset_index(drop=True)

    print("\nTest Result Summary (one-sided H1: base better, FDR-BH corrected)")
    cols = [
        "model", "n_samples", "alignment_shift_days",
        "base_loss", "comp_loss", "diff_mean", "GW_stat",
        "p_one(H1:base_better)", "p_fdr", "is_significant",
        "lags", "direction_by_t"
    ]
    formatters = {
        "base_loss": "{:.6f}".format,
        "comp_loss": "{:.6f}".format,
        "diff_mean": "{:.6f}".format,
        "GW_stat": "{:.4f}".format,
        "p_one(H1:base_better)": "{:.6e}".format,
        "p_fdr": "{:.6e}".format,
    }
    print(res_df[cols].to_string(index=False, formatters=formatters))

    out_csv = "daily_significance_results_gw_fixed.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nTable saved to: {out_csv}")

    plot_results(res_df.rename(columns={"p_one(H1:base_better)": "p_one"}), base_model)


if __name__ == "__main__":
    main()


"""
       model  n_samples  alignment_shift_days   base_loss   comp_loss   diff_mean GW_stat p_one(H1:base_better)        p_fdr  is_significant  lags direction_by_t
     dlinear        443                     0 2073.259563 2353.813176  280.553613  3.5296          2.080815e-04 2.774420e-04            True     5    base_better
     tsmixer        443                     0 2073.259563 2424.019372  350.759809  3.4283          3.036676e-04 3.036676e-04            True     5    base_better
itransformer        443                     0 2073.259563 2463.804849  390.545286  3.5556          1.885356e-04 2.774420e-04            True     5    base_better
    patchtst        443                     0 2073.259563 3741.660236 1668.400672  5.5789          1.209882e-08 4.839528e-08            True     5    base_better

"""