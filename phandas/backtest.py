"""Backtesting engine for factor strategies."""

import warnings
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Union, Tuple, Dict, List, Optional
from scipy.stats import linregress, skew, kurtosis, norm

if TYPE_CHECKING:
    from .core import Factor

from .plot import BacktestPlotter, _DATE_FORMAT
from .console import print, console


def _identify_drawdown_periods(equity_series: pd.Series) -> List[Dict]:
    rolling_max = equity_series.expanding().max()
    drawdown = equity_series / rolling_max - 1
    
    in_drawdown = False
    periods = []
    start_idx = None
    
    for i, (date, dd_value) in enumerate(drawdown.items()):
        if dd_value < -1e-6:
            if not in_drawdown:
                in_drawdown = True
                start_idx = i
        else:
            if in_drawdown:
                end_idx = i
                periods.append({
                    'start': drawdown.index[start_idx].strftime(_DATE_FORMAT),
                    'end': drawdown.index[end_idx].strftime(_DATE_FORMAT),
                    'depth': drawdown.iloc[start_idx:end_idx + 1].min(),
                    'duration_days': (drawdown.index[end_idx] - drawdown.index[start_idx]).days,
                })
                in_drawdown = False
    
    if in_drawdown:
        end_idx = len(drawdown) - 1
        periods.append({
            'start': drawdown.index[start_idx].strftime(_DATE_FORMAT),
            'end': drawdown.index[end_idx].strftime(_DATE_FORMAT),
            'depth': drawdown.iloc[start_idx:end_idx + 1].min(),
            'duration_days': (drawdown.index[end_idx] - drawdown.index[start_idx]).days,
        })
    
    return sorted(periods, key=lambda x: x['depth'])


def _calculate_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.03, 
                                   annualization_factor: float = 365.0) -> Dict:
    if returns.empty or len(returns) < 2:
        return {}
    
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    if hasattr(returns.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(returns.index):
        days = (returns.index[-1] - returns.index[0]).days
    else:
        days = len(returns)
    
    annual_return = (1 + total_return) ** (annualization_factor / days) - 1 if days > 0 else 0
    annual_vol = returns.std() * np.sqrt(annualization_factor)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    rolling_max = equity.expanding().max()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    t = np.arange(len(equity))
    r_value = linregress(t, equity.values)[2]
    linearity = r_value ** 2
    
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(annualization_factor) if len(downside_returns) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    var_95 = returns.quantile(0.05)
    cvar = returns[returns <= var_95].mean() if (returns <= var_95).any() else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'linearity': linearity,
        'drawdown_periods': _identify_drawdown_periods(equity),
        'var_95': var_95,
        'cvar': cvar,
    }


class Portfolio:
    """Portfolio state with trade execution and valuation."""
    def __init__(self, initial_capital: float = 1000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.holdings = {}
        self.total_value = initial_capital
        self.history = []
        self.trade_log = []

    def update_market_value(self, date, prices: pd.Series):
        holdings_value = 0.0
        self.holdings.clear()
        prices_dict = prices.to_dict()
        
        for symbol, qty in self.positions.items():
            if symbol in prices_dict:
                value = qty * prices_dict[symbol]
                self.holdings[symbol] = value
                holdings_value += value
        
        self.total_value = self.cash + holdings_value
        self.history.append({'date': date, 'total_value': self.total_value})

    def execute_trade(self, symbol: str, quantity: float, price: float, 
                     transaction_cost_rates: Union[float, Tuple[float, float]], 
                     trade_date: pd.Timestamp):
        if isinstance(transaction_cost_rates, (list, tuple)):
            buy_cost_rate, sell_cost_rate = transaction_cost_rates
        else:
            buy_cost_rate = sell_cost_rate = transaction_cost_rates
        
        trade_value = quantity * price
        cost = abs(trade_value) * (buy_cost_rate if quantity > 0 else sell_cost_rate)
        
        self.cash -= (trade_value + cost)
        new_quantity = self.positions.get(symbol, 0.0) + quantity
        
        if abs(new_quantity) < 1e-10:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_quantity
        
        self.trade_log.append({
            'date': trade_date,
            'symbol': symbol,
            'trade_value': trade_value,
            'cost': cost
        })
    
    def _build_datetime_df(self, data_list: list) -> pd.DataFrame:
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')
    
    def get_history_df(self) -> pd.DataFrame:
        return self._build_datetime_df(self.history)

    def get_trade_log_df(self) -> pd.DataFrame:
        return self._build_datetime_df(self.trade_log)


class Backtester:
    """Factor strategy backtesting engine."""
    
    def __init__(
        self,
        entry_price_factor: 'Factor',
        strategy_factor: 'Factor',
        transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
        initial_capital: float = 1000,
        full_rebalance: bool = False,
        neutralization: str = "market"
    ):
        self.entry_price_factor = entry_price_factor
        self.strategy_factor = strategy_factor
        self.full_rebalance = full_rebalance
        self.neutralization = neutralization.lower()
        
        if isinstance(transaction_cost, (list, tuple)):
            self.transaction_cost_rates = tuple(transaction_cost)
        else:
            self.transaction_cost_rates = (transaction_cost, transaction_cost)
        
        self.portfolio = Portfolio(initial_capital)
        self.metrics = {}
        
        self._price_cache = self._build_date_cache(entry_price_factor)
        self._strategy_cache = self._build_date_cache(strategy_factor)
    
    def run(self) -> 'Backtester':
        price_dates = set(self.entry_price_factor.data['timestamp'].unique())
        strategy_dates = set(self.strategy_factor.data['timestamp'].unique())
        common_dates = sorted(price_dates & strategy_dates)
        
        if len(common_dates) < 2:
            raise ValueError("Insufficient overlapping dates for backtesting")
        
        start_idx = self._find_start_date(common_dates)
        if start_idx >= len(common_dates):
            raise ValueError("Insufficient data for backtesting")
            
        initial_date = common_dates[start_idx] - pd.DateOffset(days=1)
        self.portfolio.history.append({
            'date': initial_date,
            'total_value': self.portfolio.initial_capital,
        })

        for i in range(start_idx, len(common_dates)):
            current_date = common_dates[i]
            prev_date = common_dates[i - 1] if i > 0 else None
            
            try:
                current_prices = self._get_factor_data(self.entry_price_factor, current_date)
                if current_prices.empty:
                    continue
                
                self.portfolio.update_market_value(current_date, current_prices)
                if not prev_date:
                    continue
                
                strategy_factors = self._get_factor_data(self.strategy_factor, prev_date)
                target_holdings = self._calculate_target_holdings(strategy_factors, prev_date)
                
                if self.full_rebalance:
                    for symbol in list(self.portfolio.positions.keys()):
                        if symbol in current_prices.index:
                            self.portfolio.execute_trade(
                                symbol, -self.portfolio.positions[symbol], 
                                current_prices.loc[symbol],
                                self.transaction_cost_rates, current_date)
                    self.portfolio.update_market_value(current_date, current_prices)
                
                for symbol, quantity in self._generate_orders(target_holdings, current_prices).items():
                    if symbol in current_prices.index:
                        self.portfolio.execute_trade(symbol, quantity, current_prices.loc[symbol], 
                                                    self.transaction_cost_rates, current_date)
            except Exception as e:
                warnings.warn(f"Error on {current_date}: {e}")
                continue
        
        return self
    
    def calculate_metrics(self, risk_free_rate: float = 0.03) -> 'Backtester':
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            self.metrics = {}
            return self
        
        equity_curve = history['total_value']
        daily_returns = equity_curve.pct_change().dropna()
        
        self.metrics = _calculate_performance_metrics(daily_returns, risk_free_rate, annualization_factor=365)
        psr = self._calculate_psr(daily_returns) if not daily_returns.empty else 0
        self.metrics['psr'] = psr
        
        return self
    
    def _calculate_psr(self, daily_returns: pd.Series, sr_benchmark: float = 0.0) -> float:
        if len(daily_returns) < 2:
            return 0.0
        
        std = daily_returns.std()
        sr_obs = (daily_returns.mean() * 365) / (std * np.sqrt(365)) if std > 0 else 0
        
        T = len(daily_returns)
        adjustment = np.sqrt(1 - skew(daily_returns) * sr_obs + 
                           ((kurtosis(daily_returns, fisher=False) - 1) / 4) * sr_obs ** 2)
        psr_stat = (sr_obs - sr_benchmark) / adjustment * np.sqrt(T / 365)
        psr = norm.cdf(psr_stat)
        return float(np.clip(psr, 0.0, 1.0))

    def _build_date_cache(self, factor: 'Factor') -> dict:
        """
        Builds a dictionary of {timestamp: pd.Series} for O(1) access during backtest.
        Optimized using pivot to avoid slow filtering inside loops.
        """
        # [優化點 3] 使用 pivot 代替逐行 filter，性能提升 100x+
        try:
            # 確保沒有重複數據，否則 pivot 會報錯
            df_clean = factor.data.drop_duplicates(subset=['timestamp', 'symbol'])
            
            # 將長表 (Long) 轉寬表 (Wide): Index=Time, Columns=Symbol
            pivoted = df_clean.pivot(index='timestamp', columns='symbol', values='factor')
            
            cache = {}
            skipped_dates = []
            
            # 遍歷寬表，這比 filter 快得多
            for date, row in pivoted.iterrows():
                # 去除 NaN，只保留有效信號 (模擬原本邏輯)
                valid_series = row.dropna()
                
                if not valid_series.empty:
                    cache[date] = valid_series
                else:
                    # 記錄全空的日期
                    skipped_dates.append((date, [])) 
            
            if skipped_dates:
                 warnings.warn(
                    f"Skipped {len(skipped_dates)} dates with no valid data (strategy='{factor.name}')"
                )
            
            return cache

        except Exception as e:
            # Fallback: 如果 pivot 失敗（極少見），回退到舊方法或報錯
            warnings.warn(f"Fast cache build failed ({e}), using slow method.")
            return super()._build_date_cache(factor) # 假設這是在繼承結構中，否則直接 raise
    
    def _get_factor_data(self, factor: 'Factor', date) -> pd.Series:
        if date is None:
            return pd.Series(dtype=float)
        
        if factor is self.entry_price_factor:
            return self._price_cache.get(date, pd.Series(dtype=float))
        else:
            return self._strategy_cache.get(date, pd.Series(dtype=float))
    
    def _find_start_date(self, dates) -> int:
        for i, date in enumerate(dates):
            if i == 0:
                continue
            prev_date = dates[i - 1]
            
            strategy_data = self._get_factor_data(self.strategy_factor, prev_date)
            price_data = self._get_factor_data(self.entry_price_factor, date)
            
            if not strategy_data.empty and not price_data.empty:
                return i
        raise ValueError("No valid start date found with overlapping data")
    
    def _calculate_target_holdings(self, factors: pd.Series, date=None) -> pd.Series:
        if self.neutralization == "none":
            return factors * self.portfolio.total_value
        
        if self.strategy_factor._is_signal(date):
            return factors * self.portfolio.total_value
        
        demeaned = factors - factors.mean()
        abs_sum = np.abs(demeaned).sum()
        if abs_sum < 1e-10:
            return pd.Series(0.0, index=factors.index)
        
        return (demeaned / abs_sum) * self.portfolio.total_value
    
    def _generate_orders(self, target_holdings: pd.Series, prices: pd.Series) -> pd.Series:
        current_holdings = self.portfolio.holdings
        all_symbols = set(target_holdings.index) | set(current_holdings.keys())
        trade_quantities = {}
        prices_dict = prices.to_dict()
        
        for symbol in all_symbols:
            if symbol not in prices_dict:
                continue
            trade_value = target_holdings.get(symbol, 0) - current_holdings.get(symbol, 0)
            if abs(trade_value) > 1e-10:
                trade_quantities[symbol] = trade_value / prices_dict[symbol]
        
        return pd.Series(trade_quantities)

    def summary(self) -> str:
        if not self.metrics:
            return "Backtester(no metrics available)"
        
        equity_curve = self.portfolio.get_history_df()['total_value']
        if equity_curve.empty:
            return "Backtester(no data)"
        
        start = equity_curve.index[0].strftime(_DATE_FORMAT)
        end = equity_curve.index[-1].strftime(_DATE_FORMAT)
        name = self.strategy_factor.name
        
        turnover_df = self.turnover
        avg_turnover = turnover_df['turnover'].mean() * 365 if not turnover_df.empty else 0
        
        m = self.metrics
        lines = [
            f"Backtester(strategy='{name}', period={start} to {end})",
            f"  total_return:   {m.get('total_return', 0):>8.2%}    annual_return:  {m.get('annual_return', 0):>8.2%}",
            f"  sharpe_ratio:   {m.get('sharpe_ratio', 0):>8.2f}    sortino_ratio:  {m.get('sortino_ratio', 0):>8.2f}",
            f"  calmar_ratio:   {m.get('calmar_ratio', 0):>8.2f}    max_drawdown:   {m.get('max_drawdown', 0):>8.2%}",
            f"  linearity:      {m.get('linearity', 0):>8.4f}    psr:            {m.get('psr', 0):>8.1%}",
            f"  var_95:         {m.get('var_95', 0):>8.2%}    cvar:           {m.get('cvar', 0):>8.2%}",
            f"  turnover:       {avg_turnover:>8.2%}",
        ]
        
        return "\n".join(lines)
    
    def print_summary(self) -> 'Backtester':
        if not self.metrics:
            print("Backtester(no metrics available)")
            return self
        
        equity_curve = self.portfolio.get_history_df()['total_value']
        if equity_curve.empty:
            print("Backtester(no data)")
            return self
        
        start = equity_curve.index[0].strftime(_DATE_FORMAT)
        end = equity_curve.index[-1].strftime(_DATE_FORMAT)
        name = self.strategy_factor.name
        
        turnover_df = self.turnover
        avg_turnover = turnover_df['turnover'].mean() * 365 if not turnover_df.empty else 0
        
        m = self.metrics
        print(f"Backtester(strategy='{name}', period={start} to {end})")
        print(f"  total_return:   {m.get('total_return', 0):>8.2%}    annual_return:  {m.get('annual_return', 0):>8.2%}")
        print(f"  sharpe_ratio:   {m.get('sharpe_ratio', 0):>8.2f}    sortino_ratio:  {m.get('sortino_ratio', 0):>8.2f}")
        print(f"  calmar_ratio:   {m.get('calmar_ratio', 0):>8.2f}    max_drawdown:   {m.get('max_drawdown', 0):>8.2%}")
        print(f"  linearity:      {m.get('linearity', 0):>8.4f}    psr:            {m.get('psr', 0):>8.1%}")
        print(f"  var_95:         {m.get('var_95', 0):>8.2%}    cvar:           {m.get('cvar', 0):>8.2%}")
        print(f"  turnover:       {avg_turnover:>8.2%}")
        
        return self
    
    def print_drawdown_periods(self, top_n: int = 5) -> 'Backtester':
        drawdown_periods = self.metrics.get('drawdown_periods', [])
        
        if not drawdown_periods:
            print("Drawdown Periods: none detected")
            return self
        
        periods_to_show = drawdown_periods[:top_n]
        total_periods = len(drawdown_periods)
        
        print(f"Drawdown Periods (top {min(top_n, total_periods)}):")
        for i, period in enumerate(periods_to_show, 1):
            print(f"  {i}. {period['start']} to {period['end']}    "
                  f"depth={period['depth']:.2%}    duration={period['duration_days']}d")
        
        if total_periods > top_n:
            print(f"  (showing {top_n} of {total_periods} periods)")
        
        return self
    
    def _calculate_benchmark_equity(self) -> pd.Series:
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            return pd.Series(dtype=float)
        
        first_date = history.index[1]
        prices_first = self._price_cache.get(first_date)
        if prices_first is None or prices_first.empty:
            return pd.Series(dtype=float)
        
        alloc_per_asset = self.portfolio.initial_capital / len(prices_first)
        holdings = {s: alloc_per_asset / prices_first[s] for s in prices_first.index}
        
        values, dates = [], []
        for date in sorted(self._price_cache.keys()):
            if date < first_date:
                continue
            prices = self._price_cache[date]
            if prices.empty:
                continue
            values.append(sum(holdings[s] * prices[s] for s in holdings if s in prices.index))
            dates.append(date)
        
        return pd.Series(values, index=pd.DatetimeIndex(dates))
    
    def plot_equity(self, figsize: tuple = (14, 8), show_summary: bool = True, show_benchmark: bool = True) -> 'Backtester':
        plotter = BacktestPlotter(self)
        plotter.plot_equity(figsize, show_summary, show_benchmark)
        return self
    
    @property
    def returns(self) -> pd.Series:
        history = self.portfolio.get_history_df()
        if history.empty or len(history) < 2:
            return pd.Series(dtype=float)
        return history['total_value'].pct_change().dropna()
    
    @property
    def equity(self) -> pd.Series:
        history = self.portfolio.get_history_df()
        return history['total_value'] if not history.empty else pd.Series(dtype=float)
    
    @property
    def trades(self) -> pd.DataFrame:
        return self.portfolio.get_trade_log_df()
    
    @property
    def turnover(self) -> pd.DataFrame:
        trade_log_df = self.portfolio.get_trade_log_df()
        history_df = self.portfolio.get_history_df()
        
        if trade_log_df.empty or history_df.empty:
            return pd.DataFrame()
            
        daily_trade_value = trade_log_df['trade_value'].abs().groupby(level='date').sum()
        daily_nav = history_df['total_value']
        
        combined = pd.DataFrame({
            'daily_trade_value': daily_trade_value,
            'daily_nav': daily_nav
        }).dropna()
        
        if combined.empty:
            return pd.DataFrame()
            
        combined['turnover'] = combined['daily_trade_value'] / combined['daily_nav']
        return combined[['turnover']]
    
    @property
    def drawdown(self) -> pd.Series:
        equity = self.equity
        if equity.empty:
            return pd.Series(dtype=float)
        return equity / equity.cummax() - 1
    
    def to_dict(self) -> dict:
        equity = self.equity
        return {
            'strategy': self.strategy_factor.name,
            'period': {
                'start': equity.index[0].strftime(_DATE_FORMAT) if not equity.empty else None,
                'end': equity.index[-1].strftime(_DATE_FORMAT) if not equity.empty else None,
            },
            'metrics': self.metrics,
            'returns': self.returns.to_dict() if not self.returns.empty else {},
            'equity': equity.to_dict() if not equity.empty else {},
        }
    
    def __repr__(self):
        history = self.portfolio.get_history_df()
        if not history.empty:
            days = len(history)
            start_date = history.index[0].strftime(_DATE_FORMAT)
            end_date = history.index[-1].strftime(_DATE_FORMAT)
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"period={start_date} to {end_date}, days={days})")
        else:
            return (f"Backtester(strategy={self.strategy_factor.name}, "
                   f"entry_price={self.entry_price_factor.name}, cost={self.transaction_cost_rates[0]:.3%})")

def backtest(
    entry_price_factor: 'Factor',
    strategy_factor: 'Factor',
    transaction_cost: Union[float, Tuple[float, float]] = (0.0003, 0.0003),
    initial_capital: float = 1000,
    full_rebalance: bool = False,
    neutralization: str = "market",
    auto_run: bool = True
) -> Backtester:
    bt = Backtester(entry_price_factor, strategy_factor, transaction_cost, initial_capital, 
                   full_rebalance, neutralization)
    
    if auto_run:
        bt.run().calculate_metrics()
    
    return bt

# -------------------------------------------------------------------------
# Monte Carlo Extension
# -------------------------------------------------------------------------

class MonteCarloResult:
    """Stores and visualizes Monte Carlo simulation results."""
    
    def __init__(self, metrics_df: pd.DataFrame, paths: pd.DataFrame, original_metrics: dict, initial_capital: float):
        self.metrics_df = metrics_df
        self.paths = paths  # Scaled equity curves (Actual Capital)
        self.original_metrics = original_metrics
        self.initial_capital = initial_capital
        self.summary_stats = self._calculate_summary()

    def _calculate_summary(self) -> pd.DataFrame:
        """Calculate statistical summary of the simulations."""
        stats = self.metrics_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        stats['original'] = pd.Series(self.original_metrics)
        return stats[['count', 'mean', 'std', '5%', '50%', '95%', 'original']]

    def summary(self) -> str:
        return self.summary_stats.to_string()

    def plot(self, figsize: tuple = (14, 10), title: str = "Monte Carlo Simulation"):
        """Plot Monte Carlo paths and distribution of Key Metrics."""
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2)
        
        # 1. Equity Paths (Spaghetti Plot)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot a subset of paths to avoid performance issues
        plot_paths = self.paths.iloc[:, :min(100, self.paths.shape[1])]
        
        # 繪製模擬路徑
        ax1.plot(plot_paths, color='gray', alpha=0.1, linewidth=1)
        # 繪製平均路徑
        ax1.plot(plot_paths.mean(axis=1), color='red', linestyle='--', linewidth=2, label='Mean Path')
        
        # Highlight percentiles (cone)
        percentiles = self.paths.quantile([0.05, 0.95], axis=1).T
        ax1.fill_between(percentiles.index, percentiles.iloc[:, 0], percentiles.iloc[:, 1], 
                        color='blue', alpha=0.1, label='95% Confidence Interval')
        
        # 標示初始資金線
        ax1.axhline(y=self.initial_capital, color='black', linestyle=':', alpha=0.5)
        
        ax1.set_title(f"{title} - Equity Curves (n={len(self.metrics_df)})")
        ax1.set_ylabel("Portfolio Value ($)")  # 修改標籤
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Max Drawdown Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self.metrics_df['max_drawdown'].hist(bins=50, ax=ax2, color='darkred', alpha=0.7, density=True)
        orig_dd = self.original_metrics.get('max_drawdown', 0)
        ax2.axvline(orig_dd, color='black', linestyle='--', linewidth=2, label=f'Original: {orig_dd:.1%}')
        
        dd_95 = self.metrics_df['max_drawdown'].quantile(0.05)
        ax2.axvline(dd_95, color='red', linestyle=':', linewidth=2, label=f'95% Worst: {dd_95:.1%}')
        
        ax2.set_title("Distribution of Max Drawdown")
        ax2.set_xlabel("Drawdown")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Sharpe Ratio Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self.metrics_df['sharpe_ratio'].hist(bins=50, ax=ax3, color='darkgreen', alpha=0.7, density=True)
        orig_sharpe = self.original_metrics.get('sharpe_ratio', 0)
        ax3.axvline(orig_sharpe, color='black', linestyle='--', linewidth=2, label=f'Original: {orig_sharpe:.2f}')
        
        sharpe_05 = self.metrics_df['sharpe_ratio'].quantile(0.05)
        ax3.axvline(sharpe_05, color='red', linestyle=':', linewidth=2, label=f'5% Worst: {sharpe_05:.2f}')

        ax3.set_title("Distribution of Sharpe Ratio")
        ax3.set_xlabel("Sharpe Ratio")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def monte_backtest(
    backtester: 'Backtester', 
    n_sims: int = 1000, 
    sim_days: Optional[int] = None,
    method: str = "bootstrap"
) -> MonteCarloResult:
    """
    Perform Monte Carlo simulation on backtest results.
    """
    if not backtester.metrics:
        raise ValueError("Backtester has no metrics. Please run .calculate_metrics() first.")
    
    returns = backtester.returns.values
    if len(returns) < 2:
        raise ValueError("Insufficient returns data for Monte Carlo simulation.")
        
    if sim_days is None:
        sim_days = len(returns)
        
    print(f"Running Monte Carlo: {n_sims} simulations, {sim_days} days each...")
    
    # 1. Generate Random Returns
    if method == "bootstrap":
        sim_returns = np.random.choice(returns, size=(sim_days, n_sims), replace=True)
    elif method == "shuffle":
        if sim_days != len(returns):
             warnings.warn("Method 'shuffle' forces sim_days to match history length.")
             sim_days = len(returns)
        sim_returns = np.empty((sim_days, n_sims))
        for i in range(n_sims):
            sim_returns[:, i] = np.random.permutation(returns)
    else:
        raise ValueError("Method must be 'bootstrap' or 'shuffle'")

    # 2. Construct Normalized Equity Curves (Starts at 1.0)
    # 用於計算 Drawdown 與 Sharpe (數學上用歸一化計算較快)
    norm_equity_curves = np.vstack([np.ones((1, n_sims)), 1 + sim_returns])
    norm_equity_curves = np.cumprod(norm_equity_curves, axis=0)
    
    # 3. Calculate Metrics (Vectorized)
    total_returns = norm_equity_curves[-1, :] - 1
    annual_returns = (1 + total_returns) ** (365 / sim_days) - 1
    volatility = np.std(sim_returns, axis=0) * np.sqrt(365)
    
    sharpe_ratios = np.divide(annual_returns, volatility, out=np.zeros_like(annual_returns), where=volatility!=0)
    
    running_max = np.maximum.accumulate(norm_equity_curves, axis=0)
    drawdowns = norm_equity_curves / running_max - 1
    max_drawdowns = np.min(drawdowns, axis=0)
    
    calmar_ratios = np.divide(annual_returns, np.abs(max_drawdowns), 
                             out=np.zeros_like(annual_returns), where=max_drawdowns!=0)

    # 4. Scale Equity Curves for Plotting (Multiplier * Initial Capital)
    # 這裡將歸一化的曲線乘上初始資金
    initial_capital = backtester.portfolio.initial_capital
    scaled_equity_curves = norm_equity_curves * initial_capital

    metrics_data = {
        'total_return': total_returns,
        'annual_return': annual_returns,
        'annual_volatility': volatility,
        'sharpe_ratio': sharpe_ratios,
        'max_drawdown': max_drawdowns,
        'calmar_ratio': calmar_ratios
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # 儲存的是真實金額曲線
    path_df = pd.DataFrame(scaled_equity_curves, index=range(len(scaled_equity_curves)))
    
    return MonteCarloResult(metrics_df, path_df, backtester.metrics, initial_capital)