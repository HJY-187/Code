# DENet-Driven R-Breaker Strategy for Futures Trading

This repository contains the official implementation of the **DENet** (Decomposition-Enhanced Network) model, a novel forecasting architecture designed to drive the **R-breaker strategy** for enhanced profitability in futures trading.

The project evaluates the model's performance across five different futures contracts using both daily and minute-level data.

---

## Project Overview

Traditional trading strategies often rely on static price levels. Our approach introduces **DENet**, a prediction model that dynamicizes the R-breaker strategy. By forecasting price movements, DENet allows the strategy to preemptively adjust entry and exit points, significantly improving risk-adjusted returns.

### Key Contributions:
- **DENet Architecture**: A specialized deep learning model for time-series forecasting in futures markets.
- **Hybrid Strategy**: Integration of DENet predictions with the classic R-breaker logic (`Advance.py`).
- **Multi-Granularity Analysis**: Comprehensive testing on both daily and minute-level frequencies.
- **Significance Testing**: Statistical validation of the trading results to ensure robustness.

---

## Repository Structure

```text
.
├── Final_Experiment/               # Core experimental code
│   ├── [I]/           # Experiments for specific futures (e.g., I,JM,J)
│   │   ├── forecasting/            # Prediction module
│   │   │   ├── daily.py            # Daily-line forecasting logic
│   │   │   ├── min.py              # Minute-line forecasting logic
│   │   │   └── results/            # Saved prediction outputs
│   │   |└── strategy/               # Trading strategy module
│   │   |     ├── r-breaker.py        # Baseline R-breaker strategy
│   │   ...   └── Advance.py          # DENet-driven (prediction-enhanced) strategy
│   └── ...                         # (Repeat for other 4 futures)
├── significance_analysis_daily/    # Statistical significance testing (Daily)
│   └── Significance_Analysis.py
├── Significance_Analysis_min/      # Statistical significance testing (Minute)
│   └── Significance_Analysis.py
├── draw/                           # Visualization scripts for paper figures
└── README.md                       # Project documentation
```
## Environment Setup
The framework is implemented using Python 3.12.3. To ensure reproducibility, please use the following environment:
### Hardware Requirements
GPU: NVIDIA GeForce RTX 4050 Laptop GPU (or higher)
### Software Requirements
CUDA: 12.1
PyTorch,2.5.1,Deep learning model construction
NumPy,1.26.4,Numerical computations
Pandas,2.2.2,Data manipulation and preprocessing
Scikit-learn,1.5.1,Data standardization and evaluation
