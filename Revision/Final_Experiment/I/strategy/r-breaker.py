import datetime
import logging
import argparse
import backtrader as bt
import numpy as np
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Data
# ============================================================
def load_data(filepath):
    """Minute level data"""
    df = pd.read_csv(
        filepath,
        parse_dates=['datetime'],
        usecols=['datetime', 'high', 'low', 'open', 'close', 'volume']
    )
    df = df.rename(columns={
        'high': 'high',
        'low': 'low',
        'open': 'open',
        'close': 'close',
        'volume': 'volume'
    })
    df = df.set_index('datetime').sort_index()
    df['trade_date'] = df.index.date
    df = df[df['close'] > 0]
    return df


# ============================================================
# Backtrader Data Feed
# ============================================================
class RBreakerData(bt.feeds.PandasData):
    lines = (
        'breakout_buy_price', 'observe_sell_price', 'reverse_sell_price',
        'reverse_buy_price', 'observe_buy_price', 'breakout_sell_price',
        'prev_high', 'prev_low', 'prev_close',
        'pred_next_close', 'pred_trend', 'pred_avg_1hr'
    )
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('breakout_buy_price', 'breakout_buy_price'),
        ('observe_sell_price', 'observe_sell_price'),
        ('reverse_sell_price', 'reverse_sell_price'),
        ('reverse_buy_price', 'reverse_buy_price'),
        ('observe_buy_price', 'observe_buy_price'),
        ('breakout_sell_price', 'breakout_sell_price'),
        ('prev_high', 'prev_high'),
        ('prev_low', 'prev_low'),
        ('prev_close', 'prev_close'),
        ('pred_next_close', 'pred_next_close'),
        ('pred_trend', 'pred_trend'),
        ('pred_avg_1hr', 'pred_avg_1hr'),
    )


# ============================================================
# Preprocess
# ============================================================
def preprocess_data(df, f1, f2, f3, prev_daily_path, daily_pred_path=None, minute_pred_path=None):
    df = df.sort_index()
    prev_daily = pd.read_csv(prev_daily_path, parse_dates=['datetime'])
    prev_daily = prev_daily.rename(columns={
        'datetime': 'cur_date',
        'high': 'prev_high',
        'low': 'prev_low',
        'close': 'prev_close'
    })
    prev_daily['cur_date'] = pd.to_datetime(prev_daily['cur_date']).dt.date
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
    merged = pd.merge(
        df.reset_index(),
        prev_daily,
        left_on='trade_date',
        right_on='cur_date',
        how='left'
    )
    merged = merged.dropna(subset=['prev_high', 'prev_low', 'prev_close'])
    merged['observe_sell_price'] = merged['prev_high'] + f1 * (merged['prev_close'] - merged['prev_low'])
    merged['observe_buy_price'] = merged['prev_low'] - f1 * (merged['prev_high'] - merged['prev_close'])
    merged['reverse_sell_price'] = (1 + f2) / 2 * (merged['prev_high'] + merged['prev_close']) - f2 * merged['prev_low']
    merged['reverse_buy_price'] = (1 + f2) / 2 * (merged['prev_low'] + merged['prev_close']) - f2 * merged['prev_high']
    merged['breakout_buy_price'] = merged['observe_sell_price'] + f3 * (
            merged['observe_sell_price'] - merged['observe_buy_price']
    )
    merged['breakout_sell_price'] = merged['observe_buy_price'] - f3 * (
            merged['observe_sell_price'] - merged['observe_buy_price']
    )
    # Keep prediction fields for compatibility with Advance.py
    merged['pred_trend_min'] = 0.0
    merged['pred_trend_daily'] = 0.0
    merged['pred_trend'] = 0.0
    # Minute predictions
    if minute_pred_path:
        print(f"Loading minute predictions from {minute_pred_path}...")
        pred_min = pd.read_csv(minute_pred_path, parse_dates=['datetime'])
        if 'pred_avg_1hr' in pred_min.columns:
            merged = merged.merge(pred_min[['datetime', 'pred_avg_1hr']], on='datetime', how='left')
            merged['pred_trend_min'] = np.sign(merged['pred_avg_1hr'] - merged['close']).fillna(0.0)
        elif 'pred_slope' in pred_min.columns:
            merged = merged.merge(pred_min[['datetime', 'pred_slope']], on='datetime', how='left')
            merged['pred_trend_min'] = np.sign(merged['pred_slope']).fillna(0.0)
        elif any(c.startswith('pred_step_') for c in pred_min.columns):
            step_cols = [c for c in pred_min.columns if c.startswith('pred_step_')]
            pred_min['pred_avg_1hr'] = pred_min[step_cols].mean(axis=1)
            merged = merged.merge(pred_min[['datetime', 'pred_avg_1hr']], on='datetime', how='left')
            merged['pred_trend_min'] = np.sign(merged['pred_avg_1hr'] - merged['close']).fillna(0.0)
        elif 'pred_close' in pred_min.columns:
            merged = merged.merge(pred_min[['datetime', 'pred_close']], on='datetime', how='left')
            merged['pred_next_close'] = merged['pred_close']
            merged['pred_trend_min'] = np.sign(merged['pred_close'] - merged['close']).fillna(0.0)
    # Daily predictions
    if daily_pred_path:
        print(f"Loading daily predictions from {daily_pred_path}...")
        merged['trade_date'] = pd.to_datetime(merged['trade_date'])
        pred_daily = pd.read_csv(daily_pred_path, parse_dates=['datetime'])
        pred_daily = pred_daily.rename(columns={'datetime': 'pred_date'})
        if 'pred_day_1' in pred_daily.columns:
            temp_merged = pd.merge(
                merged[['trade_date', 'prev_close']].drop_duplicates(),
                pred_daily,
                left_on='trade_date',
                right_on='pred_date',
                how='left'
            )
            up_votes = (
                    (temp_merged['pred_day_1'] > temp_merged['prev_close']).astype(int) +
                    (temp_merged['pred_day_2'] > temp_merged['prev_close']).astype(int) +
                    (temp_merged['pred_day_3'] > temp_merged['prev_close']).astype(int)
            )
            temp_merged['temp_trend_daily'] = up_votes.apply(
                lambda x: 1.0 if x >= 2 else (-1.0 if x <= 1 else 0.0)
            )
            merged = merged.merge(
                temp_merged[['pred_date', 'temp_trend_daily']],
                left_on='trade_date',
                right_on='pred_date',
                how='left'
            )
            merged['pred_trend_daily'] = merged['temp_trend_daily'].fillna(0.0)
            merged.drop(columns=['pred_date', 'temp_trend_daily'], inplace=True)
        elif 'pred_close' in pred_daily.columns or 'close' in pred_daily.columns:
            target_col = 'pred_close' if 'pred_close' in pred_daily.columns else 'close'
            pred_daily = pred_daily.rename(columns={target_col: 'pred_close_day'})
            merged = merged.merge(
                pred_daily[['pred_date', 'pred_close_day']],
                left_on='trade_date',
                right_on='pred_date',
                how='left'
            )
            merged['pred_trend_daily'] = np.sign(merged['pred_close_day'] - merged['prev_close']).fillna(0.0)
            merged.drop(columns=['pred_date', 'pred_close_day'], inplace=True)
    # Multi-scale fusion
    if minute_pred_path and daily_pred_path:
        min_trend = merged['pred_trend_min'].fillna(0.0)
        daily_trend = merged['pred_trend_daily'].fillna(0.0)
        merged['pred_trend'] = np.where(
            (min_trend != 0) & (daily_trend != 0) & (np.sign(min_trend) != np.sign(daily_trend)),
            min_trend,
            (min_trend + daily_trend) / 2.0
        )
    elif minute_pred_path:
        merged['pred_trend'] = merged['pred_trend_min']
    elif daily_pred_path:
        merged['pred_trend'] = merged['pred_trend_daily']
    else:
        merged['pred_trend'] = 0.0
    # Required compatibility columns
    required_cols = ['pred_next_close', 'pred_avg_1hr']
    for col in required_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    if 'cur_date' in merged.columns:
        merged = merged.drop(['cur_date'], axis=1)
    if 'trade_date' in merged.columns:
        merged = merged.drop(['trade_date'], axis=1)
    merged = merged.set_index('datetime')
    return merged


# ============================================================
# Strategy
# ============================================================
class ImprovedRBreakerStrategy(bt.Strategy):
    params = (
        ('max_reversals_per_day', 5),
        ('close_minutes_before_end', 5),
        ('market_close_fixed', None),
        ('debug', True),
        # keep params for compatibility, but prediction_filter won't be used
        ('use_prediction_filter', False),
        ('prediction_tolerance', 0.0),
        ('use_prediction_exit', False),
        ('prediction_exit_gap', 0.0),
        ('bull_max_reversals', 5),
        ('bear_max_reversals', 3),
        ('bull_close_minutes', 5),
        ('bear_close_minutes', 8),
        ('enable_dynamic_sizing', False),
    )

    def __init__(self):
        self.current_date = None
        self.daily_high = -np.inf
        self.daily_low = np.inf
        self.reversal_count = 0
        self.observe_condition = False
        self.open_order = None
        self.close_order = None
        self.market_close_time = None
        self.signal = None
        self.trend_flag = 0
        self.logger = logging.getLogger('ImprovedRBreaker')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fh = logging.FileHandler('ImprovedRBreaker.log', mode='w', encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)
        self.max_reversals_today = self.params.max_reversals_per_day
        self.close_minutes_today = self.params.close_minutes_before_end

    def log(self, txt, dt=None):
        if self.params.debug:
            dt = dt or self.data.datetime.datetime(0)
            print(f'[{dt}] {txt}')

    def next(self):
        current_date = self.data.datetime.date(0)
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_high = -np.inf
            self.daily_low = np.inf
            self.reversal_count = 0
            self.observe_condition = False
            self._update_market_close_time()
            self._update_trend_state()
            if self.trend_flag >= 0:
                self.max_reversals_today = self.params.bull_max_reversals
                self.close_minutes_today = self.params.bull_close_minutes
            else:
                self.max_reversals_today = self.params.bear_max_reversals
                self.close_minutes_today = self.params.bear_close_minutes
            self.log(f'New trading day started, market close time: {self.market_close_time}')
        self.daily_high = max(self.daily_high, self.data.high[0])
        self.daily_low = min(self.daily_low, self.data.low[0])
        current_time = self.data.datetime.time(0)
        if self._is_close_time(current_time) and self.position:
            self.log(f'Triggered close before market close, current position: {self.position.size}')
            self.close_order = self.close()
            return
        if self.position and self._should_exit_on_prediction():
            self.log(
                f'Prediction-based exit triggered, predicted={self.data.pred_next_close[0]:.2f}, current={self.data.close[0]:.2f}'
            )
            self.close_order = self.close()
            return
        if not (self.open_order or self.close_order):
            self._generate_signals()
            self._execute_orders()

    def _update_market_close_time(self):
        if self.params.market_close_fixed:
            self.market_close_time = self.params.market_close_fixed
            return
        current_date = self.current_date
        for i in range(len(self.data)):
            dt = self.data.datetime.date(i)
            if dt != current_date:
                self.market_close_time = self.data.datetime.time(i - 1)
                break

    def _is_close_time(self, current_time):
        if not self.market_close_time:
            return False
        close_dt = datetime.datetime.combine(self.current_date, self.market_close_time)
        current_dt = datetime.datetime.combine(self.current_date, current_time)
        delta_min = (close_dt - current_dt).total_seconds() / 60
        return 0 <= delta_min <= self.close_minutes_today

    def _update_trend_state(self):
        trend = 0
        if hasattr(self.data, 'pred_trend'):
            trend = self.data.pred_trend[0]
            if np.isnan(trend):
                trend = 0
        self.trend_flag = 1 if trend >= 0 else -1

    def _prediction_confirms(self, direction):
        # user asked not to use prediction_filter
        return True

    def _should_exit_on_prediction(self):
        if not self.params.use_prediction_exit or not hasattr(self.data, 'pred_next_close'):
            return False
        pred_val = self.data.pred_next_close[0]
        if np.isnan(pred_val):
            return False
        current_close = self.data.close[0]
        diff = pred_val - current_close
        gap = self.params.prediction_exit_gap
        if self.position.size > 0 and diff <= -gap:
            return True
        if self.position.size < 0 and diff >= gap:
            return True
        return False

    def _generate_signals(self):
        self.signal = None
        if not self.position:
            if self.data.high[0] >= self.data.breakout_buy_price[0]:
                if self._prediction_confirms('up'):
                    self.signal = 'breakout_buy'
                    self.log(
                        f'Breakout buy signal: high={self.data.high[0]:.2f} >= breakout_buy_price={self.data.breakout_buy_price[0]:.2f}'
                    )
            elif self.data.low[0] <= self.data.breakout_sell_price[0]:
                if self._prediction_confirms('down'):
                    self.signal = 'breakout_sell'
                    self.log(
                        f'Breakout sell signal: low={self.data.low[0]:.2f} <= breakout_sell_price={self.data.breakout_sell_price[0]:.2f}'
                    )
        else:
            if self.reversal_count >= self.max_reversals_today:
                self.log(f'Reversal count reached daily limit ({self.max_reversals_today})')
                return
            if self.position.size > 0:
                if self.data.high[0] > self.data.observe_sell_price[0]:
                    self.observe_condition = True
                    self.log(
                        f'Long observe condition triggered: high={self.data.high[0]:.2f} > observe_sell_price={self.data.observe_sell_price[0]:.2f}'
                    )
                if self.observe_condition and self.data.low[0] <= self.data.reverse_sell_price[0]:
                    if self._prediction_confirms('down'):
                        self.signal = 'reverse_sell'
                        self.reversal_count += 1
                        self.observe_condition = False
                        self.log(
                            f'Long reversal sell signal: low={self.data.low[0]:.2f} <= reverse_sell_price={self.data.reverse_sell_price[0]:.2f}'
                        )
            elif self.position.size < 0:
                if self.data.low[0] < self.data.observe_buy_price[0]:
                    self.observe_condition = True
                    self.log(
                        f'Short observe condition triggered: low={self.data.low[0]:.2f} < observe_buy_price={self.data.observe_buy_price[0]:.2f}'
                    )
                if self.observe_condition and self.data.high[0] >= self.data.reverse_buy_price[0]:
                    if self._prediction_confirms('up'):
                        self.signal = 'reverse_buy'
                        self.reversal_count += 1
                        self.observe_condition = False
                        self.log(
                            f'Short reversal buy signal: high={self.data.high[0]:.2f} >= reverse_buy_price={self.data.reverse_buy_price[0]:.2f}'
                        )

    def _calculate_size(self, signal_type):
        base_size = 1.0
        if not self.params.enable_dynamic_sizing or not hasattr(self.data, 'pred_trend'):
            return base_size
        trend = self.data.pred_trend[0]
        if np.isnan(trend):
            return base_size
        if signal_type in ['breakout_buy', 'reverse_buy']:
            if trend > 0.2:
                return 1.5
            elif trend < -0.2:
                return 0.5
            return 1.0
        if signal_type in ['breakout_sell', 'reverse_sell']:
            if trend < -0.2:
                return 1.5
            elif trend > 0.2:
                return 0.5
            return 1.0
        return base_size

    def _execute_orders(self):
        if self.signal is None:
            return
        size = self._calculate_size(self.signal)
        if self.signal == 'breakout_buy':
            self.log(f'Execute breakout buy, size={size}')
            self.open_order = self.buy(size=size)
        elif self.signal == 'breakout_sell':
            self.log(f'Execute breakout sell, size={size}')
            self.open_order = self.sell(size=size)
        elif self.signal == 'reverse_sell':
            self.log(f'Execute reverse sell, size={size}')
            self.close()
            self.open_order = self.sell(size=size)
        elif self.signal == 'reverse_buy':
            self.log(f'Execute reverse buy, size={size}')
            self.close()
            self.open_order = self.buy(size=size)
        self.signal = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order == self.close_order:
                self.log(
                    f'Close completed: price={order.executed.price:.2f}, qty={order.executed.size:.2f}'
                )
                self.close_order = None
            elif order == self.open_order:
                self.log(
                    f'Open completed: price={order.executed.price:.2f}, qty={order.executed.size:.2f}'
                )
                self.open_order = None
        else:
            self.log(f'Order not completed: {order.getstatusname()}')
            if order == self.open_order:
                self.open_order = None
            elif order == self.close_order:
                self.close_order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'Trade completed: gross={trade.pnl:.2f}, net={trade.pnlcomm:.2f}')

    def stop(self):
        self.log(f' capital: {self.broker.getvalue():.2f}')
        self.log(f' position: {self.position.size}')


# ============================================================
# Utility: print standard results
# ============================================================
def print_standard_result_block(
        total_return=0.0,
        total_trades=0,
        won_trades=0,
        max_drawdown=0.0,
        annual_return=0.0,
        sharpe_ratio=-999.0,
        calmar_ratio=-999.0
):
    win_rate = ((won_trades / total_trades) * 100.0) if total_trades > 0 else 0.0
    print("\n===== Backtest Results =====")
    print(f"Total return rate: {total_return:.2f}%")
    print(f"Total number of trades: {total_trades}")
    print(f"Trades: {total_trades}")
    print(f"Number of winning trades: {won_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Maximum drawdown: {max_drawdown:.2f}%")
    print(f"Annualized return rate: {annual_return:.2f}%")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Calmar ratio: {calmar_ratio:.2f}")


# ============================================================
# Backtest runner
# ============================================================
def run_backtest(
        f1,
        f2,
        f3,
        data_path,
        prev_daily_path,
        minute_pred_path=None,
        daily_pred_path=None,
        enable_prediction_filter=False,
        enable_dynamic_sizing=False,
        prediction_tolerance=0.0,
        prediction_exit_gap=0.0,
        initial_capital=100000,
        commission=0.0001,
        slippage_perc=0.0,
        mult=10,
        market_close_time=None,
        start_date=None,
        end_date=None
):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(
        commission=commission,
        mult=mult,
        commtype=bt.CommInfoBase.COMM_PERC
    )
    # 1Tick
    cerebro.broker.set_slippage_fixed(
        fixed=0.5,
        slip_open=True,
        slip_limit=True,
        slip_match=True,
        slip_out=False
    )
    raw_df = load_data(data_path)
    if start_date is not None:
        raw_df = raw_df[raw_df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        raw_df = raw_df[raw_df.index < pd.to_datetime(end_date)]
    if raw_df.empty:
        print("Data preprocessing failed: no valid raw data in date range.")
        print_standard_result_block(
            total_return=0.0,
            total_trades=0,
            won_trades=0,
            max_drawdown=0.0,
            annual_return=0.0,
            sharpe_ratio=-999.0,
            calmar_ratio=-999.0
        )
        return
    processed_df = preprocess_data(
        raw_df,
        f1=f1,
        f2=f2,
        f3=f3,
        prev_daily_path=prev_daily_path,
        daily_pred_path=daily_pred_path,
        minute_pred_path=minute_pred_path
    )
    if processed_df.empty:
        print("Data preprocessing failed: no valid processed data.")
        print_standard_result_block(
            total_return=0.0,
            total_trades=0,
            won_trades=0,
            max_drawdown=0.0,
            annual_return=0.0,
            sharpe_ratio=-999.0,
            calmar_ratio=-999.0
        )
        return
    data = RBreakerData(dataname=processed_df)
    cerebro.adddata(data)
    cerebro.addstrategy(
        ImprovedRBreakerStrategy,
        market_close_fixed=market_close_time,
        use_prediction_filter=False,  # force off
        enable_dynamic_sizing=enable_dynamic_sizing,
        prediction_tolerance=prediction_tolerance,
        use_prediction_exit=(prediction_exit_gap > 0),
        prediction_exit_gap=prediction_exit_gap
    )
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name='sharpe',
        riskfreerate=0.01771,
        timeframe=bt.TimeFrame.Days,
        annualize=True,
        factor=252
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)
    initial_cash = cerebro.broker.getvalue()
    print(f'Initial capital: {initial_cash:.2f}')
    print(f'Commission rate: {commission:.6f}')

    try:
        results = cerebro.run()
        final_cash = cerebro.broker.getvalue()
        strategy = results[0]
        total_return = ((final_cash - initial_cash) / initial_cash * 100.0)
        trade_analysis = strategy.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('closed', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        max_drawdown = getattr(drawdown_analysis.max, 'drawdown', 0.0) if hasattr(drawdown_analysis, 'max') else 0.0
        returns = strategy.analyzers.returns.get_analysis()
        annual_return = returns.get('rnorm100', 0.0)
        sharpe = strategy.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe.get('sharperatio', None)
        if sharpe_ratio is None or (
                isinstance(sharpe_ratio, float) and (np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio))):
            sharpe_ratio = -999.0

        calmar_ratio = (annual_return / max_drawdown) if max_drawdown > 0 else -999.0
        print_standard_result_block(
            total_return=total_return,
            total_trades=total_trades,
            won_trades=won_trades,
            max_drawdown=max_drawdown,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio
        )
    except Exception as e:
        print(f"Backtest execution failed: {e}")
        print_standard_result_block(
            total_return=0.0,
            total_trades=0,
            won_trades=0,
            max_drawdown=0.0,
            annual_return=0.0,
            sharpe_ratio=-999.0,
            calmar_ratio=-999.0
        )


# ============================================================
# Entry
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the improved R-Breaker backtest.')
    parser.add_argument(
        '--minute-data-path',
        default='I_5.csv',
        help='Path to minute-level market data CSV, default I_5.csv'
    )
    parser.add_argument(
        '--prev-daily-path',
        default='Previous_Daily_Data.csv',
        help="Path to previous day's daily bar CSV"
    )
    parser.add_argument(
        '--daily-pred-path',
        default=None,
        help='Path to daily prediction CSV'
    )
    parser.add_argument(
        '--minute-pred-path',
        default=None,
        help='Path to minute prediction CSV'
    )
    parser.add_argument(
        '--enable-prediction-filter',
        action='store_true',
        help='Kept only for compatibility; ignored in this version'
    )
    parser.add_argument(
        '--enable-dynamic-sizing',
        action='store_true',
        help='Enable dynamic position sizing'
    )
    parser.add_argument('--f1', type=float, default=0.8, help='f1 parameter')
    parser.add_argument('--f2', type=float, default=0.3, help='f2 parameter')
    parser.add_argument('--f3', type=float, default=0.2, help='f3 parameter')
    parser.add_argument(
        '--prediction-tolerance',
        type=float,
        default=0.0,
        help='Kept for compatibility'
    )
    parser.add_argument(
        '--prediction-exit-gap',
        type=float,
        default=0.0,
        help='Prediction exit gap'
    )
    parser.add_argument('--start-date', default=None, help='Backtest start date YYYY-MM-DD')
    parser.add_argument('--end-date', default=None, help='Backtest end date YYYY-MM-DD')
    # not use
    parser.add_argument(
        '--slippage-perc',
        type=float,
        default=0.0000,
        help='Slippage percentage, e.g. 0.0005 = 0.05%'
    )
    args = parser.parse_args()
    mult = 10
    initial_capital = 13650
    commission = 0.0001
    market_close_time = datetime.time(15, 0)
    run_backtest(
        f1=args.f1,
        f2=args.f2,
        f3=args.f3,
        data_path=args.minute_data_path,
        prev_daily_path=args.prev_daily_path,
        minute_pred_path=args.minute_pred_path,
        daily_pred_path=args.daily_pred_path,
        enable_prediction_filter=False,  # force off
        enable_dynamic_sizing=args.enable_dynamic_sizing,
        prediction_tolerance=args.prediction_tolerance,
        prediction_exit_gap=args.prediction_exit_gap,
        initial_capital=initial_capital,
        commission=commission,
        slippage_perc=args.slippage_perc,
        mult=mult,
        market_close_time=market_close_time,
        start_date=args.start_date,
        end_date=args.end_date
    )