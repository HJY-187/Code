import datetime
import logging
import argparse
import backtrader as bt
import numpy as np
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ========== data ==========
def load_data(filepath):
    """Minute level data"""
    df = pd.read_csv(
        filepath,
        parse_dates=['datetime'],
        usecols=['datetime', 'high', 'low', 'open', 'close', 'volume']
    )

    df.rename(columns={
        'high': 'high',
        'low': 'low',
        'open': 'open',
        'close': 'close',
        'volume': 'volume'
    }, inplace=True)
    df = df.set_index('datetime').sort_index()
    df['trade_date'] = df.index.date
    df = df[df['close'] > 0]
    return df


# ========== computer ==========
class RBreakerData(bt.feeds.PandasData):
    lines = (
        'breakout_buy_price', 'observe_sell_price', 'reverse_sell_price', 'reverse_buy_price', 'observe_buy_price',
        'breakout_sell_price',
        'prev_high', 'prev_low', 'prev_close', 'pred_next_close', 'pred_trend', 'pred_avg_1hr'
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


def preprocess_data(df, f1, f2, f3, prev_daily_path, daily_pred_path=None, minute_pred_path=None):
    df = df.sort_index()

    prev_daily = pd.read_csv(
        prev_daily_path,
        parse_dates=['datetime']
    )

    prev_daily = prev_daily.rename(columns={
        'datetime': 'cur_date',
        'high': 'prev_high',
        'low': 'prev_low',
        'close': 'prev_close'
    })
    prev_daily['cur_date'] = pd.to_datetime(prev_daily['cur_date']).dt.date

    df['trade_date'] = df['trade_date'].astype('datetime64[ns]').dt.date
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
            merged['observe_sell_price'] - merged['observe_buy_price'])
    merged['breakout_sell_price'] = merged['observe_buy_price'] - f3 * (
            merged['observe_sell_price'] - merged['observe_buy_price'])

    merged['pred_trend_min'] = 0.0
    merged['pred_trend_daily'] = 0.0

    # 1. Load minute-level predictions
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
            merged['pred_trend_min'] = np.sign(merged['pred_close'] - merged['close']).fillna(0.0)

    # 2. Load daily-level predictions
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
            up_votes = (temp_merged['pred_day_1'] > temp_merged['prev_close']).astype(int) + \
                       (temp_merged['pred_day_2'] > temp_merged['prev_close']).astype(int) + \
                       (temp_merged['pred_day_3'] > temp_merged['prev_close']).astype(int)
            temp_merged['temp_trend_daily'] = up_votes.apply(lambda x: 1.0 if x >= 2 else (-1.0 if x <= 1 else 0.0))
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

    # 3. Multi-scale Fusion
    if minute_pred_path and daily_pred_path:
        merged['pred_trend'] = (merged['pred_trend_min'] + merged['pred_trend_daily']) / 2.0
    elif minute_pred_path:
        merged['pred_trend'] = merged['pred_trend_min']
    elif daily_pred_path:
        merged['pred_trend'] = merged['pred_trend_daily']
    else:
        merged['pred_trend'] = 0.0

    # 4. Ensure all columns required by RBreakerData class exist
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


# ========== Strategy Logic ==========
class ImprovedRBreakerStrategy(bt.Strategy):
    """Improved R-Breaker Strategy"""
    params = (
        ('max_reversals_per_day', 5),
        ('close_minutes_before_end', 5),
        ('market_close_fixed', None),
        ('debug', True),
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

        self.logger = logging.getLogger('ImprovedRBreaker')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fh = logging.FileHandler('../../J/strategy/ImprovedRBreaker.log', mode='w')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)

        self.max_reversals_today = self.params.max_reversals_per_day
        self.close_minutes_today = self.params.close_minutes_before_end
        self.trend_flag = 0

        # 仅新增：每日净值记录（用于正确计算夏普）
        self.daily_values = []
        self.last_date = None

    def log(self, txt, dt=None):
        if self.params.debug:
            dt = dt or self.data.datetime.datetime(0)
            print(f'[{dt}] {txt}')

    def next(self):
        current_date = self.data.datetime.date(0)
        # 仅新增：每日收盘记录净值
        if current_date != self.last_date:
            if self.last_date is not None:
                self.daily_values.append(self.broker.getvalue())
            self.last_date = current_date

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
                f'Prediction-based take profit/stop loss triggered, predicted price={self.data.pred_next_close[0]:.2f}, current price={self.data.close[0]:.2f}')
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
        if not self.params.use_prediction_filter or not hasattr(self.data, 'pred_next_close'):
            return True
        pred_val = self.data.pred_next_close[0]
        if np.isnan(pred_val):
            return False
        current_close = self.data.close[0]
        tol = self.params.prediction_tolerance
        if direction == 'up':
            return pred_val >= current_close + tol
        return pred_val <= current_close - tol

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
                        f'Breakout buy signal triggered: high={self.data.high[0]:.2f} ≥ breakout price={self.data.breakout_buy_price[0]:.2f}')
                else:
                    self.log('Prediction filter: Breakout buy rejected')
            elif self.data.low[0] <= self.data.breakout_sell_price[0]:
                if self._prediction_confirms('down'):
                    self.signal = 'breakout_sell'
                    self.log(
                        f'Breakout sell signal triggered: low={self.data.low[0]:.2f} ≤ breakout price={self.data.breakout_sell_price[0]:.2f}')
                else:
                    self.log('Prediction filter: Breakout sell rejected')

        else:
            if self.reversal_count >= self.max_reversals_today:
                self.log(f'Reversal count reached daily limit({self.max_reversals_today})')
                return

            if self.position.size > 0:
                if self.data.high[0] > self.data.observe_sell_price[0]:
                    self.observe_condition = True
                    self.log(
                        f'Triggered long position observation condition: high={self.data.high[0]:.2f} > observe sell price={self.data.observe_sell_price[0]:.2f}')
                if self.observe_condition and self.data.low[0] <= self.data.reverse_sell_price[0]:
                    if self._prediction_confirms('down'):
                        self.signal = 'reverse_sell'
                        self.log(
                            f'Triggered long position reversal signal: low={self.data.low[0]:.2f} ≤ reverse sell price={self.data.reverse_sell_price[0]:.2f}')
                        self.reversal_count += 1
                        self.observe_condition = False
                    else:
                        self.log('Prediction filter: Long position reversal rejected')
            elif self.position.size < 0:
                if self.data.low[0] < self.data.observe_buy_price[0]:
                    self.observe_condition = True
                    self.log(
                        f'Triggered short position observation condition: low={self.data.low[0]:.2f} < observe buy price={self.data.observe_buy_price[0]:.2f}')
                if self.observe_condition and self.data.high[0] >= self.data.reverse_buy_price[0]:
                    if self._prediction_confirms('up'):
                        self.signal = 'reverse_buy'
                        self.log(
                            f'Triggered short position reversal signal: high={self.data.high[0]:.2f} ≥ reverse buy price={self.data.reverse_buy_price[0]:.2f}')
                        self.reversal_count += 1
                        self.observe_condition = False
                    else:
                        self.log('Prediction filter: Short position reversal rejected')

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
            else:
                return 1.0

        if signal_type in ['breakout_sell', 'reverse_sell']:
            if trend < -0.2:
                return 1.5
            elif trend > 0.2:
                return 0.5
            else:
                return 1.0

        return base_size

    def _execute_orders(self):
        size = self._calculate_size(self.signal)

        if self.signal == 'breakout_buy':
            self.log(f'Execute breakout buy, position size: {size}')
            self.open_order = self.buy(size=size)
        elif self.signal == 'breakout_sell':
            self.log(f'Execute breakout sell, position size: {size}')
            self.open_order = self.sell(size=size)
        elif self.signal == 'reverse_sell':
            self.log(f'Execute reversal sell, position size: {size}')
            self.close()
            self.open_order = self.sell(size=size)
        elif self.signal == 'reverse_buy':
            self.log(f'Execute reversal buy, position size: {size}')
            self.close()
            self.open_order = self.buy(size=size)
        self.signal = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order == self.close_order:
                self.log(
                    f'Position closed completed: price={order.executed.price:.2f}, quantity={order.executed.size:.0f}')
                self.close_order = None
                if hasattr(self, 'pending_signal'):
                    if self.pending_signal == 'buy':
                        self.open_order = self.buy()
                    elif self.pending_signal == 'sell':
                        self.open_order = self.sell()
                    delattr(self, 'pending_signal')
            elif order == self.open_order:
                self.log(
                    f'Position opened completed: price={order.executed.price:.2f}, quantity={order.executed.size:.0f}')
                self.open_order = None
        else:
            self.log(f'Order not completed: {order.getstatusname()}, type={"Buy" if order.isbuy() else "Sell"}')
            if order == self.open_order:
                self.open_order = None
            elif order == self.close_order:
                self.close_order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'Trade completed: gross profit={trade.pnl:.2f}, net profit={trade.pnlcomm:.2f}')

    def stop(self):
        # 仅新增：记录最后一天净值
        self.daily_values.append(self.broker.getvalue())
        self.log(f' capital: {self.broker.getvalue():.2f}')
        self.log(f' position: {self.position.size}')


# ========== Backtest Execution ==========
def run_backtest(f1, f2, f3, data_path, prev_daily_path, minute_pred_path=None, daily_pred_path=None,
                 enable_prediction_filter=False, enable_dynamic_sizing=False, prediction_tolerance=0.0,
                 prediction_exit_gap=0.0, initial_capital=100000, commission=0.0001, slippage_perc=0.0, mult=10,
                 market_close_time=None, start_date=None, end_date=None):
    """Backtest execution function"""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)

    # 手续费
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
        print("Data preprocessing failed: no valid data")
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
        print("Data preprocessing failed: no valid data")
        return

    data = RBreakerData(dataname=processed_df)
    cerebro.adddata(data)
    cerebro.addstrategy(
        ImprovedRBreakerStrategy,
        market_close_fixed=market_close_time,
        use_prediction_filter=enable_prediction_filter and bool(minute_pred_path),
        enable_dynamic_sizing=enable_dynamic_sizing,
        prediction_tolerance=prediction_tolerance,
        use_prediction_exit=(prediction_exit_gap > 0),
        prediction_exit_gap=prediction_exit_gap
    )

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)
    # Backtrader
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.02)

    initial_cash = cerebro.broker.getvalue()
    print(f'Initial capital: {initial_cash:.2f}')
    print(f'Commission rate: {commission:.6f}')

    results = cerebro.run()
    final_cash = cerebro.broker.getvalue()
    strategy = results[0]

    print("\n===== Backtest Results =====")
    total_return = (final_cash - initial_cash) / initial_cash * 100
    print(f'Total return rate: {total_return:.2f}%')

    trade_analysis = strategy.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('closed', 0)
    won_trades = trade_analysis.get('won', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
    print(f'Total number of trades: {total_trades}')
    print(f'Number of winning trades: {won_trades}')
    print(f'Win rate: {win_rate:.2f}%')

    drawdown = strategy.analyzers.drawdown.get_analysis().max.drawdown
    print(f'Maximum drawdown: {drawdown:.2f}%')

    returns = strategy.analyzers.returns.get_analysis()
    annual_return = returns.get('rnorm100', 0)
    print(f'Annualized return rate: {annual_return:.2f}%')

    # ========== 核心修改：使用 Backtrader 内置年化参数计算夏普 ==========
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get('sharperatio')
    if sharpe_ratio is None:
        sharpe_ratio = 0.0
    print(f'Sharpe ratio: {sharpe_ratio:.2f}')

    # ========== Calmar ==========
    calmar_ratio = annual_return / abs(drawdown) if drawdown != 0 else 0
    print(f'Calmar ratio: {calmar_ratio:.2f}')


# ========== Entry Function ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the improved R-Breaker backtest.')
    parser.add_argument(
        '--minute-data-path',
        default='J_5.csv',
        help='Path to minute-level market data CSV, default J_5.csv'
    )
    parser.add_argument(
        '--prev-daily-path',
        default='Previous_Daily_Data.csv',
        help='Path to previous day\'s daily bar CSV for calculating R-Breaker levels'
    )
    parser.add_argument(
        '--daily-pred-path',
        default=None,
        help='Path to daily prediction CSV (for market condition classification), default None'
    )
    parser.add_argument(
        '--minute-pred-path',
        default=None,
        help='Path to 5-minute prediction CSV (contains current_datetime,pred_close) for signal filtering'
    )
    parser.add_argument(
        '--enable-prediction-filter',
        action='store_true',
        help='Enable prediction direction filter, only effective when minute-pred-path is provided'
    )
    parser.add_argument(
        '--enable-dynamic-sizing',
        action='store_true',
        help='Enable dynamic position sizing, adjust position size based on prediction'
    )
    parser.add_argument('--f1', type=float, default=0.8, help='f1 parameter (default: 0.8)')
    parser.add_argument('--f2', type=float, default=0.3, help='f2 parameter (default: 0.3)')
    parser.add_argument('--f3', type=float, default=0.2, help='f3 parameter (default: 0.2)')

    parser.add_argument(
        '--prediction-tolerance',
        type=float,
        default=0.0,
        help='Prediction tolerance (e.g., 0.2 means long only if predicted increase > 0.2%)'
    )
    parser.add_argument(
        '--prediction-exit-gap',
        type=float,
        default=0.0,
        help='Gap threshold between predicted price and current price, consider taking profit if gap narrows within this value'
    )

    parser.add_argument('--start-date', default=None, help='Backtest start date, format YYYY-MM-DD')
    parser.add_argument('--end-date', default=None, help='Backtest end date, format YYYY-MM-DD')


    parser.add_argument( #not use
        '--slippage-perc',
        type=float,
        default=0.0005,
        help='Slippage percentage, e.g. 0.0005 means 0.05%'
    )

    args = parser.parse_args()

    mult = 10
    initial_capital = 90000
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
        enable_prediction_filter=args.enable_prediction_filter,
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
Annualized return rate: 1.11%
Total return rate: 0.70%
Sharpe ratio: -0.01
Win rate: 69.57%
Maximum drawdown: 8.43%
Total trades: 23

[Prediction-Enhanced R-Breaker]
Annualized return rate: 7.65%
Total return rate: 4.76%
Sharpe ratio: 0.58
Win rate: 69.57%
Maximum drawdown: 8.81%
Total trades: 23
"""
