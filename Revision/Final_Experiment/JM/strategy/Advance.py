import pandas as pd
import numpy as np
import subprocess
import re
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
import argparse
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Config
F1_RANGE = np.arange(0.2, 0.9, 0.1)
F2_RANGE = np.arange(0.1, 0.5, 0.1)
F3_RANGE = np.arange(0.1, 0.5, 0.1)

MAX_WORKERS = 4
SCRIPT_PATH = "r-breaker.py"


def parse_metrics(output):
    annual_ret_match = re.search(r"Annualized return rate:\s*(-?\d+\.?\d*)%", output)
    total_ret_match = re.search(r"Total return rate:\s*(-?\d+\.?\d*)%", output)
    win_match = re.search(r"Win rate:\s*(-?\d+\.?\d*)%", output)
    dd_match = re.search(r"Maximum drawdown:\s*(-?\d+\.?\d*)%", output)
    sharpe_match = re.search(r"Sharpe ratio:\s*(-?\d+\.?\d*)", output)
    trades_match = re.search(r"Total number of trades:\s*(\d+)", output)

    annual_return = float(annual_ret_match.group(1)) if annual_ret_match else -999.0
    total_return = float(total_ret_match.group(1)) if total_ret_match else -999.0
    win_rate = float(win_match.group(1)) if win_match else 0.0
    max_dd = float(dd_match.group(1)) if dd_match else 100.0
    sharpe_ratio = float(sharpe_match.group(1)) if sharpe_match else -999.0
    total_trades = int(trades_match.group(1)) if trades_match else 0

    return {
        "annual_return": annual_return,
        "total_return": total_return,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe_ratio,
        "total_trades": total_trades
    }


def run_backtest(
    f1, f2, f3,
    start_date, end_date,
    minute_data_path,
    prev_daily_path,
    daily_pred=None,
    minute_pred=None,
    enable_dynamic_sizing=False,
    enable_prediction_filter=False,
    slippage_perc=0.0
):
    """
    Call r-breaker.py and parse output
    """
    cmd = [
        sys.executable, SCRIPT_PATH,
        "--f1", str(round(f1, 2)),
        "--f2", str(round(f2, 2)),
        "--f3", str(round(f3, 2)),
        "--minute-data-path", minute_data_path,
        "--prev-daily-path", prev_daily_path,
        "--start-date", start_date,
        "--end-date", end_date,
        "--slippage-perc", str(slippage_perc)
    ]

    if enable_dynamic_sizing:
        cmd.append("--enable-dynamic-sizing")

    if enable_prediction_filter:
        cmd.append("--enable-prediction-filter")

    if daily_pred is not None:
        cmd.extend(["--daily-pred-path", daily_pred])

    if minute_pred is not None:
        cmd.extend(["--minute-pred-path", minute_pred])

    try:
        result = subprocess.run(cmd, capture_output=True)

        try:
            output = result.stdout.decode('utf-8')
        except UnicodeDecodeError:
            try:
                output = result.stdout.decode('gbk')
            except Exception:
                output = str(result.stdout)

        metrics = parse_metrics(output)
        return metrics

    except Exception:
        return {
            "annual_return": -999.0,
            "total_return": -999.0,
            "win_rate": 0.0,
            "max_drawdown": 100.0,
            "sharpe_ratio": -999.0,
            "total_trades": 0
        }


def optimize_on_prediction_period(
    model_name,
    train_start,
    train_end,
    minute_data_path,
    prev_daily_path,
    daily_pred,
    minute_pred,
    slippage_perc=0.0
):
    print(f"\n=======================================================")
    print(f"Starting Parameter Optimization for Model: {model_name}")
    print(f"Optimization Period (Prediction Active): {train_start} to {train_end}")
    print(f"Daily Prediction File: {daily_pred}")
    print(f"Minute Prediction File: {minute_pred}")

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
            executor.submit(
                run_backtest,
                f1, f2, f3,
                train_start, train_end,
                minute_data_path,
                prev_daily_path,
                daily_pred,   # use daily prediction
                minute_pred,  # use minute prediction
                True,         # keep same as enhanced strategy
                False,        # keep unchanged
                slippage_perc
            ): (f1, f2, f3)
            for f1, f2, f3 in combinations
        }

        count = 0
        for future in as_completed(future_to_params):
            f1, f2, f3 = future_to_params[future]

            metrics = future.result()
            annual_ret = metrics["annual_return"]
            total_ret = metrics["total_return"]
            win = metrics["win_rate"]
            dd = metrics["max_drawdown"]
            sharpe = metrics["sharpe_ratio"]
            trades = metrics["total_trades"]

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
                'max_drawdown': dd,
                'sharpe_ratio': sharpe,
                'total_trades': trades
            })

            if annual_ret > best_annual_return:
                best_annual_return = annual_ret
                best_total_return = total_ret
                best_params = (f1, f2, f3)
                best_metrics = metrics

                print(
                    f"*** New Best for {model_name}: "
                    f"Annualized return rate {annual_ret:.2f}% | "
                    f"Total return rate {total_ret:.2f}% | "
                    f"Sharpe {sharpe:.2f} | "
                    f"Win: {win:.2f}% | DD: {dd:.2f}% | "
                    f"Trades: {trades} "
                    f"[f1={f1:.2f}, f2={f2:.2f}, f3={f3:.2f}] ***"
                )

    end_time = time.time()

    print("\n" + "=" * 60)
    print(f"Optimization Completed for {model_name}")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    if best_params:
        print(f"Best Annual Return: {best_annual_return:.2f}%")
        print(f"Best Total Return: {best_total_return:.2f}%")
        print(f"Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {best_metrics['win_rate']:.2f}%")
        print(f"Max Drawdown: {best_metrics['max_drawdown']:.2f}%")
        print(f"Total Trades: {best_metrics['total_trades']}")
        print(f"Best Parameters: f1={best_params[0]:.2f}, f2={best_params[1]:.2f}, f3={best_params[2]:.2f}")
    else:
        print("No valid results found.")
    print("=" * 60)

    df = pd.DataFrame(results)
    df = df.sort_values('annual_return', ascending=False)
    filename = f'opt_results_prediction_train_{model_name}.csv'
    df.to_csv(filename, index=False)
    print(f"Detailed optimization results saved to {filename}")

    return best_params, best_metrics


def compare_strategies_on_prediction_period(
    model_name,
    best_params,
    eval_start,
    eval_end,
    minute_data_path,
    prev_daily_path,
    daily_pred,
    minute_pred,
    slippage_perc=0.0
):
    f1, f2, f3 = best_params

    print(f"\n=======================================================")
    print(f"Out-of-Sample Comparison for Model: {model_name}")
    print(f"Evaluation Period (Prediction Active): {eval_start} to {eval_end}")
    print(f"Fixed Parameters from Training: f1={f1:.2f}, f2={f2:.2f}, f3={f3:.2f}")
    print(f"Slippage: {slippage_perc}")
    print(f"=======================================================")

    print("\nRunning Original R-Breaker (No Prediction)...")
    original_metrics = run_backtest(
        f1, f2, f3,
        eval_start, eval_end,
        minute_data_path,
        prev_daily_path,
        daily_pred=None,
        minute_pred=None,
        enable_dynamic_sizing=False,
        enable_prediction_filter=False,
        slippage_perc=slippage_perc
    )

    print("\nRunning Prediction-Enhanced R-Breaker...")
    enhanced_metrics = run_backtest(
        f1, f2, f3,
        eval_start, eval_end,
        minute_data_path,
        prev_daily_path,
        daily_pred=daily_pred,
        minute_pred=minute_pred,
        enable_dynamic_sizing=True,
        enable_prediction_filter=False,
        slippage_perc=slippage_perc
    )

    print("\n" + "=" * 60)
    print("Out-of-Sample Comparison Results")
    print("=" * 60)

    print("\n[Original R-Breaker]")
    print(f"Annualized return rate: {original_metrics['annual_return']:.2f}%")
    print(f"Total return rate: {original_metrics['total_return']:.2f}%")
    print(f"Sharpe ratio: {original_metrics['sharpe_ratio']:.2f}")
    print(f"Win rate: {original_metrics['win_rate']:.2f}%")
    print(f"Maximum drawdown: {original_metrics['max_drawdown']:.2f}%")
    print(f"Total trades: {original_metrics['total_trades']}")

    print("\n[Prediction-Enhanced R-Breaker]")
    print(f"Annualized return rate: {enhanced_metrics['annual_return']:.2f}%")
    print(f"Total return rate: {enhanced_metrics['total_return']:.2f}%")
    print(f"Sharpe ratio: {enhanced_metrics['sharpe_ratio']:.2f}")
    print(f"Win rate: {enhanced_metrics['win_rate']:.2f}%")
    print(f"Maximum drawdown: {enhanced_metrics['max_drawdown']:.2f}%")
    print(f"Total trades: {enhanced_metrics['total_trades']}")

    comparison_df = pd.DataFrame([
        {
            "strategy": "Original_RBreaker",
            "period_start": eval_start,
            "period_end": eval_end,
            "f1": round(f1, 2),
            "f2": round(f2, 2),
            "f3": round(f3, 2),
            "slippage_perc": slippage_perc,
            "annual_return": original_metrics["annual_return"],
            "total_return": original_metrics["total_return"],
            "sharpe_ratio": original_metrics["sharpe_ratio"],
            "win_rate": original_metrics["win_rate"],
            "max_drawdown": original_metrics["max_drawdown"],
            "total_trades": original_metrics["total_trades"]
        },
        {
            "strategy": "Prediction_Enhanced_RBreaker",
            "period_start": eval_start,
            "period_end": eval_end,
            "f1": round(f1, 2),
            "f2": round(f2, 2),
            "f3": round(f3, 2),
            "slippage_perc": slippage_perc,
            "annual_return": enhanced_metrics["annual_return"],
            "total_return": enhanced_metrics["total_return"],
            "sharpe_ratio": enhanced_metrics["sharpe_ratio"],
            "win_rate": enhanced_metrics["win_rate"],
            "max_drawdown": enhanced_metrics["max_drawdown"],
            "total_trades": enhanced_metrics["total_trades"]
        }
    ])

    filename = f'comparison_oos_{model_name}.csv'
    comparison_df.to_csv(filename, index=False)
    print(f"\nComparison results saved to {filename}")

    return original_metrics, enhanced_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOS validation for R-Breaker parameters.")

    parser.add_argument("--minute-data-path", default="JM_5.csv")
    parser.add_argument("--prev-daily-path", default="Previous_Daily_Data.csv")
    parser.add_argument("--daily-pred-path", default="pred_daily_advanced.csv")
    parser.add_argument("--minute-pred-path", default="pred_minute_advanced.csv")

    parser.add_argument("--train-start", default="2023-5-16")
    parser.add_argument("--train-end", default="2024-5-9")

    parser.add_argument("--eval-start", default="2024-5-10")
    parser.add_argument("--eval-end", default="2024-12-31")

    parser.add_argument("--slippage-perc", type=float, default=0.0005,
                        help="Slippage percentage, e.g. 0.0005 means 0.05%")

    args = parser.parse_args()

    model_name = "Advanced_DENet"

    best_params, best_metrics = optimize_on_prediction_period(
        model_name=model_name,
        train_start=args.train_start,
        train_end=args.train_end,
        minute_data_path=args.minute_data_path,
        prev_daily_path=args.prev_daily_path,
        daily_pred=args.daily_pred_path,
        minute_pred=args.minute_pred_path,
        slippage_perc=args.slippage_perc
    )

    if best_params is None:
        print("No valid best parameters found in training period.")
        sys.exit(0)

    compare_strategies_on_prediction_period(
        model_name=model_name,
        best_params=best_params,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        minute_data_path=args.minute_data_path,
        prev_daily_path=args.prev_daily_path,
        daily_pred=args.daily_pred_path,
        minute_pred=args.minute_pred_path,
        slippage_perc=args.slippage_perc
    )

"""
=======================================================
Out-of-Sample Comparison for Model: Advanced_DENet
Evaluation Period (Prediction Active): 2024-5-10 to 2024-12-31
Fixed Parameters from Training: f1=0.60, f2=0.30, f3=0.10
Slippage: 0.0005
=======================================================

Running Original R-Breaker (No Prediction)...

Running Prediction-Enhanced R-Breaker...

============================================================
Out-of-Sample Comparison Results
============================================================

[Original R-Breaker]
Annualized return rate: 1.41%
Total return rate: 0.88%
Sharpe ratio: 0.02
Win rate: 69.57%
Maximum drawdown: 8.39%
Total trades: 23

[Prediction-Enhanced R-Breaker]
Annualized return rate: 7.96%
Total return rate: 4.95%
Sharpe ratio: 0.61
Win rate: 69.57%
Maximum drawdown: 8.77%
Total trades: 23

"""
