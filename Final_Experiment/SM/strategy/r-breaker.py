import datetime
import logging
import argparse
import backtrader as bt
import numpy as np
import pandas as pd


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
    prev_daily['cur_date'] = pd.to_datetime(prev_daily['cur_date']).dt.date  # Convert to date format (without time)

    # Merge minute data with previous day's daily data (matched by trading day)
    df['trade_date'] = df['trade_date'].astype('datetime64[ns]').dt.date  # Ensure format consistency
    merged = pd.merge(
        df.reset_index(),  # Minute data: contains datetime (index), trade_date (current day's date)
        prev_daily,
        left_on='trade_date',  # Left table: current trading day (e.g., 2023-01-04)
        right_on='cur_date',  # Right table: previous trading day (e.g., 2023-01-03)
        how='left'
    )

    # Filter records without previous day's data (avoid indicator calculation errors)
    merged = merged.dropna(subset=['prev_high', 'prev_low', 'prev_close'])

    # Calculate R-Breaker key prices (core formula)
    merged['observe_sell_price'] = merged['prev_high'] + f1 * (merged['prev_close'] - merged['prev_low'])
    merged['observe_buy_price'] = merged['prev_low'] - f1 * (merged['prev_high'] - merged['prev_close'])
    merged['reverse_sell_price'] = (1 + f2) / 2 * (merged['prev_high'] + merged['prev_close']) - f2 * merged['prev_low']
    merged['reverse_buy_price'] = (1 + f2) / 2 * (merged['prev_low'] + merged['prev_close']) - f2 * merged['prev_high']
    merged['breakout_buy_price'] = merged['observe_sell_price'] + f3 * (
                merged['observe_sell_price'] - merged['observe_buy_price'])
    merged['breakout_sell_price'] = merged['observe_buy_price'] - f3 * (
                merged['observe_sell_price'] - merged['observe_buy_price'])

    # Market condition prediction (optional)
    # Initialize market condition prediction
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
            merged = merged.merge(temp_merged[['pred_date', 'temp_trend_daily']], left_on='trade_date',
                                  right_on='pred_date', how='left')
            merged['pred_trend_daily'] = merged['temp_trend_daily'].fillna(0.0)
            merged.drop(columns=['pred_date', 'temp_trend_daily'], inplace=True)
        elif 'pred_close' in pred_daily.columns or 'close' in pred_daily.columns:
            target_col = 'pred_close' if 'pred_close' in pred_daily.columns else 'close'
            pred_daily = pred_daily.rename(columns={target_col: 'pred_close_day'})
            merged = merged.merge(pred_daily[['pred_date', 'pred_close_day']], left_on='trade_date',
                                  right_on='pred_date', how='left')
            merged['pred_trend_daily'] = np.sign(merged['pred_close_day'] - merged['prev_close']).fillna(0.0)
            merged.drop(columns=['pred_date', 'pred_close_day'], inplace=True)

    # 3. Multi-scale Fusion
    # If both exist, take average; if only one exists, use that; if none, 0
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

    # Organize index and columns
    if 'cur_date' in merged.columns:
        merged = merged.drop(['cur_date'], axis=1)
    if 'trade_date' in merged.columns:
        merged = merged.drop(['trade_date'], axis=1)

    merged = merged.set_index('datetime')  # Restore datetime as index
    return merged


# ========== Strategy Logic (Fix order and time logic flaws) ==========
class ImprovedRBreakerStrategy(bt.Strategy):
    """Improved R-Breaker Strategy (Fixed order status and market close time logic)"""
    params = (
        ('max_reversals_per_day', 5),  # Maximum number of reversals per day (reasonable limit to avoid over-trading)
        ('close_minutes_before_end', 5),  # Minutes to close positions before market close
        ('market_close_fixed', None),
        # Fixed market close time (e.g., datetime.time(15, 0), higher priority than auto-calculation)
        ('debug', True),  # Debug log switch
        ('use_prediction_filter', False),
        ('prediction_tolerance', 0.0),
        ('use_prediction_exit', False),
        ('prediction_exit_gap', 0.0),
        ('bull_max_reversals', 5),
        ('bear_max_reversals', 3),
        ('bull_close_minutes', 5),
        ('bear_close_minutes', 8),
        ('enable_dynamic_sizing', False),  # New parameter for dynamic sizing
    )

    def __init__(self):
        # Track intraday status
        self.current_date = None
        self.daily_high = -np.inf
        self.daily_low = np.inf
        self.reversal_count = 0
        self.observe_condition = False

        # Order status (added logic to clean up unexecuted orders)
        self.open_order = None
        self.close_order = None
        self.market_close_time = None  # Actual market close time of the day

        # Log configuration (ensure log file path is writable)
        self.logger = logging.getLogger('ImprovedRBreaker')
        self.logger.setLevel(logging.DEBUG)
        # Avoid adding handlers repeatedly
        if not self.logger.handlers:
            fh = logging.FileHandler('../../SM/Strategy/ImprovedRBreaker.log', mode='w')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)

        self.max_reversals_today = self.params.max_reversals_per_day
        self.close_minutes_today = self.params.close_minutes_before_end
        self.trend_flag = 0

    def log(self, txt, dt=None):
        """Log output (unified format)"""
        if self.params.debug:
            dt = dt or self.data.datetime.datetime(0)
            print(f'[{dt}] {txt}')

    def next(self):
        """Triggered per bar, core logic"""
        # 1. Initialize for new trading day
        current_date = self.data.datetime.date(0)
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_high = -np.inf
            self.daily_low = np.inf
            self.reversal_count = 0
            self.observe_condition = False
            self._update_market_close_time()  # Update market close time for the day
            self._update_trend_state()
            if self.trend_flag >= 0:
                self.max_reversals_today = self.params.bull_max_reversals
                self.close_minutes_today = self.params.bull_close_minutes
            else:
                self.max_reversals_today = self.params.bear_max_reversals
                self.close_minutes_today = self.params.bear_close_minutes
            self.log(f'New trading day started, market close time: {self.market_close_time}')

        # 2. Update intraday high and low prices
        self.daily_high = max(self.daily_high, self.data.high[0])
        self.daily_low = min(self.daily_low, self.data.low[0])

        # 3. Force close positions before market close (ensure trigger)
        current_time = self.data.datetime.time(0)
        if self._is_close_time(current_time) and self.position:
            self.log(f'Triggered close before market close, current position: {self.position.size}')
            self.close_order = self.close()  # Close position order
            return  # Do not execute subsequent logic after closing positions

        # 3.1 Prediction-based take profit/stop loss (does not affect signal generation)
        if self.position and self._should_exit_on_prediction():
            self.log(
                f'Prediction-based take profit/stop loss triggered, predicted price={self.data.pred_next_close[0]:.2f}, current price={self.data.close[0]:.2f}')
            self.close_order = self.close()
            return

        # 4. Generate and execute signals when there are no pending orders
        if not (self.open_order or self.close_order):
            self._generate_signals()
            self._execute_orders()

    def _update_market_close_time(self):
        """Update market close time for the day (use fixed time first, otherwise auto-calculate)"""
        if self.params.market_close_fixed:
            self.market_close_time = self.params.market_close_fixed
            return

        # Auto-calculate: Get the time of the last bar of the day (traverse data to find the last bar of the day)
        current_date = self.current_date
        # Traverse backward from current position to find the last bar of the day
        for i in range(len(self.data)):
            dt = self.data.datetime.date(i)
            if dt != current_date:
                # The previous bar is the last bar of the day
                self.market_close_time = self.data.datetime.time(i - 1)
                break

    def _is_close_time(self, current_time):
        """Determine if it's time to close positions before market close (fixed time calculation logic)"""
        if not self.market_close_time:
            return False  # Do not trigger if market close time is not obtained

        # Construct datetime objects for market close time and current time of the day
        close_dt = datetime.datetime.combine(self.current_date, self.market_close_time)
        current_dt = datetime.datetime.combine(self.current_date, current_time)
        delta_min = (close_dt - current_dt).total_seconds() / 60

        # Ensure time difference is non-negative (avoid cross-day errors)
        return 0 <= delta_min <= self.close_minutes_today

    def _update_trend_state(self):
        """Update market condition flag based on daily prediction"""
        trend = 0
        if hasattr(self.data, 'pred_trend'):
            trend = self.data.pred_trend[0]
            if np.isnan(trend):
                trend = 0
        self.trend_flag = 1 if trend >= 0 else -1

    def _prediction_confirms(self, direction):
        """Prediction filter: Determine if the predicted direction of the next 5-minute bar is consistent with the signal"""
        if not self.params.use_prediction_filter or not hasattr(self.data, 'pred_next_close'):
            return True
        pred_val = self.data.pred_next_close[0]
        if np.isnan(pred_val):
            return False  # Reject trigger when there is no prediction to avoid information inconsistency
        current_close = self.data.close[0]
        tol = self.params.prediction_tolerance
        if direction == 'up':
            return pred_val >= current_close + tol
        return pred_val <= current_close - tol

    def _should_exit_on_prediction(self):
        """Early take profit/stop loss based on predicted price of next bar"""
        if not self.params.use_prediction_exit or not hasattr(self.data, 'pred_next_close'):
            return False
        pred_val = self.data.pred_next_close[0]
        if np.isnan(pred_val):
            return False
        current_close = self.data.close[0]
        diff = pred_val - current_close
        gap = self.params.prediction_exit_gap
        if self.position.size > 0 and diff <= -gap:
            return True  # Long position: predicted to drop
        if self.position.size < 0 and diff >= gap:
            return True  # Short position: predicted to rise
        return False

    def _generate_signals(self):
        """Generate trading signals (keep core logic unchanged)"""
        self.signal = None

        # No position: breakout signals
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

        # With position: reversal signals
        else:
            if self.reversal_count >= self.max_reversals_today:
                self.log(f'Reversal count reached daily limit({self.max_reversals_today})')
                return

            if self.position.size > 0:  # Long position
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
            elif self.position.size < 0:  # Short position
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
        """
        Calculate dynamic position size based on multi-scale prediction fusion.
        MODE: Asymmetric Enhancement
        Strategy:
        - Agreement (Trend confirms Signal): 1.5x (Aggressive)
        - Disagreement (Trend opposes Signal): 0.5x (Conservative)
        - No Veto: Never return 0.0. Always respect R-Breaker's primal signal.
        """
        base_size = 1.0
        if not self.params.enable_dynamic_sizing or not hasattr(self.data, 'pred_trend'):
            return base_size

        trend = self.data.pred_trend[0]
        if np.isnan(trend):
            return base_size

        # Long signals
        if signal_type in ['breakout_buy', 'reverse_buy']:
            if trend > 0.2:  # Prediction is Bullish
                return 1.5  # Enhance
            elif trend < -0.2:  # Prediction is Bearish (Conflict)
                return 0.5  # Reduce but don't stop
            else:
                return 1.0  # Neutral/Unsure

        # Short signals
        if signal_type in ['breakout_sell', 'reverse_sell']:
            if trend < -0.2:  # Prediction is Bearish
                return 1.5  # Enhance
            elif trend > 0.2:  # Prediction is Bullish (Conflict)
                return 0.5  # Reduce but don't stop
            else:
                return 1.0  # Neutral/Unsure

        return base_size

    def _execute_orders(self):
        """Execute orders (support dynamic position sizing)"""
        size = self._calculate_size(self.signal)

        if self.signal == 'breakout_buy':
            self.log(f'Execute breakout buy, position size: {size}')
            self.open_order = self.buy(size=size)  # Open long position
        elif self.signal == 'breakout_sell':
            self.log(f'Execute breakout sell, position size: {size}')
            self.open_order = self.sell(size=size)  # Open short position
        elif self.signal == 'reverse_sell':
            self.log(f'Execute reversal sell, position size: {size}')
            self.close()  # Close long position
            self.open_order = self.sell(size=size)  # Open short position
        elif self.signal == 'reverse_buy':
            self.log(f'Execute reversal buy, position size: {size}')
            self.close()  # Close short position
            self.open_order = self.buy(size=size)  # Open long position
        self.signal = None  # Reset signal

    def notify_order(self, order):
        """Order status notification (handle all possible statuses)"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted, no action
            return

        # Order completed/rejected/canceled/insufficient margin
        if order.status == order.Completed:
            if order == self.close_order:
                self.log(
                    f'Position closed completed: price={order.executed.price:.2f}, quantity={order.executed.size:.0f}')
                self.close_order = None
                # Handle reversal signals after closing positions
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
            # Order not completed (rejected/canceled/insufficient margin), clean up order status
            self.log(f'Order not completed: {order.getstatusname()}, type={"Buy" if order.isbuy() else "Sell"}')
            if order == self.open_order:
                self.open_order = None
            elif order == self.close_order:
                self.close_order = None

    def notify_trade(self, trade):
        """Trade completion notification (record profit and loss)"""
        if not trade.isclosed:
            return
        self.log(f'Trade completed: gross profit={trade.pnl:.2f}, net profit={trade.pnlcomm:.2f}')

    def stop(self):
        """Output key indicators at the end of backtest"""
        self.log(f'Final capital: {self.broker.getvalue():.2f}')
        self.log(f'Final position: {self.position.size}')


# ========== Backtest Execution (Improve cost and indicator calculation) ==========
def run_backtest(f1, f2, f3, data_path, prev_daily_path, minute_pred_path=None, daily_pred_path=None,
                 enable_prediction_filter=False, enable_dynamic_sizing=False, prediction_tolerance=0.0,
                 prediction_exit_gap=0.0, initial_capital=100000, commission=0.0001, mult=10, market_close_time=None):
    """Backtest execution function (parameterized configuration, supports commission and fixed market close time)"""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)
    # Set transaction costs (commission rate + contract multiplier)
    cerebro.broker.setcommission(
        commission=commission,  # Commission per lot (e.g., 0.0001 means 0.01%)
        mult=mult,  # Contract multiplier
        commtype=bt.CommInfoBase.COMM_PERC
        # Calculate commission as percentage (can also be changed to COMM_FIXED for fixed fee)
    )

    # Load and preprocess data
    raw_df = load_data(data_path)
    processed_df = preprocess_data(
        raw_df,
        f1=f1,
        f2=f2,
        f3=f3,
        prev_daily_path=prev_daily_path,  # Pass previous day's daily data path from outside
        daily_pred_path=daily_pred_path,
        minute_pred_path=minute_pred_path
    )
    if processed_df.empty:
        print("Data preprocessing failed: no valid data")
        return

    # Add data and strategy (supports passing fixed market close time)
    data = RBreakerData(dataname=processed_df)
    cerebro.adddata(data)
    cerebro.addstrategy(
        ImprovedRBreakerStrategy,
        market_close_fixed=market_close_time,  # Pass fixed market close time (e.g., futures 15:00)
        use_prediction_filter=enable_prediction_filter and bool(minute_pred_path),
        enable_dynamic_sizing=enable_dynamic_sizing,
        prediction_tolerance=prediction_tolerance,
        use_prediction_exit=(prediction_exit_gap > 0),
        prediction_exit_gap=prediction_exit_gap
    )

    # Add analyzers (improve indicator calculation)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01771, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)

    # Run backtest
    initial_cash = cerebro.broker.getvalue()
    print(f'Initial capital: {initial_cash:.2f}')
    results = cerebro.run()
    final_cash = cerebro.broker.getvalue()
    strategy = results[0]

    # Extract and print backtest indicators (improve exception handling)
    print("\n===== Backtest Results =====")
    print(f'Total return rate: {((final_cash - initial_cash) / initial_cash * 100):.2f}%')

    trade_analysis = strategy.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('closed', 0)
    won_trades = trade_analysis.get('won', {}).get('total', 0)
    print(f'Total number of trades: {total_trades}')
    print(f'Number of winning trades: {won_trades}')
    print(f'Win rate: {((won_trades / total_trades) * 100) if total_trades > 0 else 0:.2f}%')

    drawdown = strategy.analyzers.drawdown.get_analysis().max.drawdown
    print(f'Maximum drawdown: {drawdown:.2f}%')

    returns = strategy.analyzers.returns.get_analysis()
    annual_return = returns.get('rnorm100', 0)  # Annualized return rate (percentage)
    print(f'Annualized return rate: {annual_return:.2f}%')

    sharpe = strategy.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get('sharperatio', 0)
    print(f'Sharpe ratio: {sharpe_ratio:.2f}')


# ========== Entry Function (Parameterized configuration, avoid hardcoding) ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the improved R-Breaker backtest.')
    parser.add_argument(
        '--minute-data-path',
        default='SM_5.csv',
        help='Path to minute-level market data CSV, default SM_5.csv'
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
    # Parameter optimization arguments
    parser.add_argument('--f1', type=float, default=0.6, help='f1 parameter (default: 0.6)')
    parser.add_argument('--f2', type=float, default=0.2, help='f2 parameter (default: 0.2)')
    parser.add_argument('--f3', type=float, default=0.1, help='f3 parameter (default: 0.1)')

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

    args = parser.parse_args()

    # Core strategy parameters (adjustable as needed)
    # f1, f2, f3 = 0.5, 0.3, 0.2  # R-Breaker indicator parameters (Optimized) # Removed as now from args
    mult = 10  # Contract multiplier (e.g., JM coke futures is 10)
    initial_capital = 4800  # Initial capital
    commission = 0.0001  # Commission rate (0.01%)
    # Fixed market close time (e.g., futures market closes at 15:00, adjust according to actual product)
    market_close_time = datetime.time(15, 0)

    # Execute backtest
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
        mult=mult,
        market_close_time=market_close_time
    )
"""
===== Backtest Results =====
Total return rate: 608.51%
Total number of trades: 81
Number of winning trades: 53
Win rate: 65.43%
Maximum drawdown: 149.69%
Annualized return rate: 175.43%
Sharpe ratio: 0.02
"""
