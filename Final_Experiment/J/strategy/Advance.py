import pandas as pd
import numpy as np
import subprocess
import re
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
import argparse

# Config
# F1, F2, F3 ranges for grid search
# Slightly wider range to ensure we catch the peak
F1_RANGE = np.arange(0.2, 0.9, 0.1)
F2_RANGE = np.arange(0.1, 0.5, 0.1)
F3_RANGE = np.arange(0.1, 0.5, 0.1)
#
# Workers
MAX_WORKERS = 4

# Script
SCRIPT_PATH = "r-breaker.py"


def run_backtest(f1, f2, f3, daily_pred, minute_pred):
    """
    Call r-breaker.py and parse output
    """
    cmd = [
        sys.executable, SCRIPT_PATH,
        "--f1", str(round(f1, 2)),
        "--f2", str(round(f2, 2)),
        "--f3", str(round(f3, 2)),
        "--enable-dynamic-sizing",
        "--daily-pred-path", daily_pred,
        "--minute-pred-path", minute_pred
    ]

    try:
        # Capture raw bytes
        result = subprocess.run(cmd, capture_output=True)

        # Decode output
        try:
            output = result.stdout.decode('utf-8')
        except UnicodeDecodeError:
            try:
                output = result.stdout.decode('gbk')
            except:
                output = str(result.stdout)

        # Regex to find different metrics in the output
        annual_ret_match = re.search(r"Annualized return rate:\s*(-?\d+\.?\d*)%", output)  # 明确命名：年化收益率
        total_ret_match = re.search(r"Total return rate:\s*(-?\d+\.?\d*)%", output)    # 新增：匹配总收益率
        win_match = re.search(r"Win rate:\s*(-?\d+\.?\d*)%", output)
        dd_match = re.search(r"Maximum drawdown:\s*(-?\d+\.?\d*)%", output)


        annual_return = float(annual_ret_match.group(1)) if annual_ret_match else -999.0

        total_return = float(total_ret_match.group(1)) if total_ret_match else -999.0
        win_rate = float(win_match.group(1)) if win_match else 0.0
        max_dd = float(dd_match.group(1)) if dd_match else 100.0


        return annual_return, total_return, win_rate, max_dd

    except Exception as e:

        return -999.0, -999.0, 0.0, 100.0


def optimize(model_name, daily_pred, minute_pred):
    print(f"\n=======================================================")
    print(f"Starting Optimization for Model: {model_name}")
    print(f"Using: {daily_pred} & {minute_pred}")
    print(f"=======================================================")

    combinations = list(itertools.product(F1_RANGE, F2_RANGE, F3_RANGE))
    print(f"Total combinations to test: {len(combinations)}")

    results = []
    best_annual_return = -100.0
    best_total_return = -100.0
    best_params = None
    best_metrics = None

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_params = {
            executor.submit(run_backtest, f1, f2, f3, daily_pred, minute_pred): (f1, f2, f3)
            for f1, f2, f3 in combinations
        }

        count = 0
        for future in as_completed(future_to_params):
            f1, f2, f3 = future_to_params[future]

            annual_ret, total_ret, win, dd = future.result()
            count += 1

            if count % 50 == 0:
                print(f"[{count}/{len(combinations)}] Current Best Annual Return: {best_annual_return:.2f}%")


            results.append({
                'f1': round(f1, 2),
                'f2': round(f2, 2),
                'f3': round(f3, 2),
                'annual_return': annual_ret,
                'total_return': total_ret,
                'win_rate': win,
                'max_drawdown': dd
            })


            if annual_ret > best_annual_return:
                best_annual_return = annual_ret
                best_total_return = total_ret
                best_params = (f1, f2, f3)
                best_metrics = (win, dd, total_ret)
                # 新增：打印时输出总收益率
                print(
                    f"*** New Best for {model_name}: Annualized return rate {annual_ret:.2f}% | Total return rate {total_ret:.2f}% (Win: {win:.2f}%, DD: {dd:.2f}%) [f1={f1:.2f}, f2={f2:.2f}, f3={f3:.2f}] ***")

    end_time = time.time()

    print("\n" + "=" * 60)
    print(f"Optimization Completed for {model_name}")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    if best_params:

        print(f"Best Annual Return: {best_annual_return:.2f}%")
        print(f"Best Total Return: {best_total_return:.2f}%")
        print(f"Win Rate: {best_metrics[0]:.2f}%")
        print(f"Max Drawdown: {best_metrics[1]:.2f}%")
        print(f"Best Parameters: f1={best_params[0]:.2f}, f2={best_params[1]:.2f}, f3={best_params[2]:.2f}")
    else:
        print("No valid results found.")
    print("=" * 60)

    # total_return
    df = pd.DataFrame(results)
    df = df.sort_values('annual_return', ascending=False)
    filename = f'opt_results_{model_name}.csv'
    df.to_csv(filename, index=False)
    print(f"Detailed results saved to {filename} ")

    return best_annual_return, best_total_return, best_metrics, best_params


if __name__ == "__main__":
    best_annual_ret_adv, best_total_ret_adv, best_metrics_adv, best_params_adv = optimize(
        "Advanced_DENet",
        "pred_daily_advanced.csv",
        "pred_minute_advanced.csv"
    )


"""
============================================================
Optimization Completed for Advanced_DENet
Time Taken: 255.69 seconds
Best Annual Return: 13.89%
Best Total Return: 28.71%
Win Rate: 67.35%
Max Drawdown: 12.39%
Best Parameters: f1=0.80, f2=0.40, f3=0.10
============================================================

"""