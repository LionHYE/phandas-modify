"""Data acquisition and management for cryptocurrency markets via CCXT."""

import warnings
import pandas as pd
import ccxt
import time
import os
from datetime import datetime, timedelta
from typing import List, Optional, TYPE_CHECKING, Callable, Dict, Any, Union

if TYPE_CHECKING:
    from .panel import Panel

from .constants import SYMBOL_RENAMES

# ==============================================================================
# 常數定義
# ==============================================================================

TIMEFRAME_MAP = {
    '1m': 'min',
    '3m': '3min',
    '5m': '5min',
    '15m': '15min',
    '30m': '30min',
    '1h': 'h',
    '2h': '2h',
    '4h': '4h',
    '6h': '6h',
    '8h': '8h',
    '12h': '12h',
    '1d': 'D',
    '3d': '3D',
    '1w': 'W',
    '1M': 'MS',
}

FETCH_BATCH_SIZE = 1000

# ==============================================================================
# 主要 API 函數
# ==============================================================================

def fetch_data(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> 'Panel':
    """Fetch, merge, and align multi-source cryptocurrency data."""
    if sources is None:
        sources = ['binance']

    return fetch_panel_core(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        sources=sources,
        output_path=output_path
    )


def fetch_panel_core(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> 'Panel':
    """核心數據獲取與整合邏輯"""
    if sources is None:
        sources = ['binance']

    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported: {list(TIMEFRAME_MAP.keys())}"
        )

    source_map = {
        'binance': fetch_binance,
        'bybit': fetch_bybit,
        'bybit_spot': lambda s, tf, sd, ed: fetch_bybit(s, tf, sd, ed, category='spot'),
        'bybit_linear': lambda s, tf, sd, ed: fetch_bybit(s, tf, sd, ed, category='linear'),
        'bybit_inverse': lambda s, tf, sd, ed: fetch_bybit(s, tf, sd, ed, category='inverse'),
        'benchmark': fetch_benchmark,
        'calendar': fetch_calendar,
        'vwap': fetch_vwap,
    }

    raw_dfs = []
    reference_end_date = None

    priority_sources = ['binance', 'bybit', 'bybit_spot', 'bybit_linear']
    sorted_sources = sorted(
        sources,
        key=lambda x: priority_sources.index(x) if x in priority_sources else 999
    )

    for source in sorted_sources:
        if source not in source_map:
            warnings.warn(f"Unknown source: {source}")
            continue

        try:
            print(f"Fetching data from {source} ({timeframe})...")

            current_end_date = end_date
            if source not in priority_sources and reference_end_date:
                current_end_date = reference_end_date

            df = source_map[source](symbols, timeframe, start_date, current_end_date)

            if df is not None and not df.empty:
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                if source in priority_sources and not reference_end_date:
                    max_ts = df['timestamp'].max()
                    if timeframe in ['1d', '1w', '1M']:
                        reference_end_date = max_ts.strftime('%Y-%m-%d')
                    else:
                        reference_end_date = max_ts.strftime('%Y-%m-%d %H:%M:%S')

                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()

                df = df.loc[:, ~df.columns.duplicated()]
                raw_dfs.append(df)
            else:
                warnings.warn(f"No data returned from {source}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to fetch from {source}: {e}")

    if not raw_dfs:
        raise ValueError("No data fetched from any source")

    combined = raw_dfs[0]
    for df in raw_dfs[1:]:
        combined = pd.merge(
            combined,
            df,
            on=['timestamp', 'symbol'],
            how='outer'
        )

    if combined.columns.duplicated().any():
        combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]

    processed = _process_data(combined, timeframe, symbols)

    int_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'dayofweek', 'is_market_hours'
    ]
    for col in int_cols:
        if col in processed.columns:
            processed[col] = processed[col].astype('Int64')

    try:
        from .panel import Panel
        result = Panel(processed)
    except ImportError:
        result = processed

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if hasattr(result, 'to_csv'):
            result.to_csv(output_path)
        else:
            result.to_csv(output_path, index=False)

    return result


# ==============================================================================
# 通用 OHLCV 獲取函數
# ==============================================================================

def _fetch_ohlcv_data(
    exchange,
    symbols: List[str],
    timeframe: str,
    since: Optional[int],
    until: Optional[int] = None,
    columns_post_process: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    market_type: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """通用 OHLCV 獲取函數"""

    def _fetch_single(sym: str) -> Optional[pd.DataFrame]:
        try:
            market_sym = f'{sym}/USDT'

            if not exchange.markets:
                exchange.load_markets()

            if 'bybit' in exchange.id.lower() and market_type:
                if market_type == 'linear':
                    market_sym = f"{sym}/USDT:USDT"
                elif market_type == 'inverse':
                    market_sym = f"{sym}/USD:{sym}"
                elif market_type == 'spot':
                    market_sym = f'{sym}/USDT'

                if market_sym not in exchange.symbols:
                    found = False
                    for s in exchange.symbols:
                        if sym in s and 'USDT' in s:
                            market_sym = s
                            found = True
                            break
                    if not found:
                        warnings.warn(f"{market_sym} not available in Bybit {market_type} market")
                        return None

            elif market_sym not in exchange.symbols:
                found = False
                for s in exchange.symbols:
                    if s == f"{sym}/USDT:USDT":
                        market_sym = s
                        found = True
                        break
                if not found:
                    warnings.warn(f"{market_sym} not available in exchange symbols")
                    return None

            all_candles = []
            cursor = since

            max_loops = 10000
            loop_count = 0

            while True:
                loop_count += 1
                if loop_count > max_loops:
                    warnings.warn(f"Loop limit reached for {sym}")
                    break

                if until and cursor and cursor > until:
                    break

                batch = exchange.fetch_ohlcv(
                    market_sym,
                    timeframe,
                    since=cursor,
                    limit=FETCH_BATCH_SIZE
                )

                if not batch:
                    break

                if until:
                    batch = [c for c in batch if c[0] <= until]
                    if not batch:
                        break

                all_candles.extend(batch)

                last_ts = batch[-1][0]

                if len(batch) < FETCH_BATCH_SIZE:
                    break

                if until and last_ts >= until:
                    break

                cursor = last_ts + 1
                time.sleep(exchange.rateLimit / 1000)

            if not all_candles:
                return None

            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = sym

            return df

        except Exception as e:
            warnings.warn(f"Failed to fetch {sym}: {e}")
            return None

    dfs = []
    for symbol in symbols:
        df = _fetch_single(symbol)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return None

    result = pd.concat(dfs, ignore_index=True)

    if columns_post_process:
        result = columns_post_process(result)

    return result


# ==============================================================================
# 各數據源的獲取函數
# ==============================================================================

def fetch_binance(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """從 Binance 獲取 OHLCV 數據"""
    try:
        exchange = ccxt.binance()
        if not exchange.has['fetchOHLCV']:
            raise RuntimeError("Binance does not support OHLCV")

        since = None
        if start_date:
            since = exchange.parse8601(
                pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ')
            )

        until = None
        if end_date:
            dt = pd.to_datetime(end_date)
            if len(str(end_date)) <= 10 and dt.hour == 0 and dt.minute == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            until = exchange.parse8601(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))

        symbols_to_fetch = list(set(symbols))

        for new_sym, rename_info in SYMBOL_RENAMES.items():
            if new_sym not in symbols_to_fetch:
                continue

            old_sym = rename_info['old_symbol']
            cutoff_date = rename_info['cutoff_date']
            cutoff_ts = exchange.parse8601(f'{cutoff_date}T00:00:00Z')

            need_old = since is None or since < cutoff_ts
            need_new = until is None or until >= cutoff_ts

            dfs_to_concat = []

            if need_old:
                old_until = min(until, cutoff_ts - 1) if until else cutoff_ts - 1
                old_data = _fetch_ohlcv_data(
                    exchange,
                    [old_sym],
                    timeframe,
                    since,
                    old_until
                )
                if old_data is not None:
                    old_data['symbol'] = new_sym
                    dfs_to_concat.append(old_data)

            if need_new:
                new_since = max(since, cutoff_ts) if since else cutoff_ts
                new_data = _fetch_ohlcv_data(
                    exchange,
                    [new_sym],
                    timeframe,
                    new_since,
                    until
                )
                if new_data is not None:
                    dfs_to_concat.append(new_data)

            other_symbols = [s for s in symbols_to_fetch if s != new_sym]
            other_data = None
            if other_symbols:
                other_data = _fetch_ohlcv_data(
                    exchange,
                    other_symbols,
                    timeframe,
                    since,
                    until
                )

            renamed_full_df = None
            if dfs_to_concat:
                renamed_full_df = pd.concat(dfs_to_concat, ignore_index=True)
                renamed_full_df = renamed_full_df.sort_values('timestamp')
                renamed_full_df = renamed_full_df.drop_duplicates(subset=['timestamp'])

                renamed_full_df = renamed_full_df.set_index('timestamp')
                freq = TIMEFRAME_MAP.get(timeframe, 'D')

                full_idx = pd.date_range(
                    start=renamed_full_df.index.min(),
                    end=renamed_full_df.index.max(),
                    freq=freq
                )
                renamed_full_df = renamed_full_df.reindex(full_idx)

                cols_ffill = ['open', 'high', 'low', 'close', 'symbol']
                renamed_full_df[cols_ffill] = renamed_full_df[cols_ffill].ffill()
                renamed_full_df['volume'] = renamed_full_df['volume'].fillna(0)

                renamed_full_df = renamed_full_df.reset_index()
                renamed_full_df = renamed_full_df.rename(columns={'index': 'timestamp'})
                renamed_full_df['symbol'] = new_sym

            final_parts = []
            if other_data is not None:
                final_parts.append(other_data)
            if renamed_full_df is not None:
                final_parts.append(renamed_full_df)

            if final_parts:
                return pd.concat(final_parts, ignore_index=True)
            else:
                return None

        return _fetch_ohlcv_data(
            exchange,
            symbols_to_fetch,
            timeframe,
            since,
            until
        )

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Binance: {e}")


def fetch_bybit(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: str = 'spot'
) -> Optional[pd.DataFrame]:
    """從 Bybit 獲取 OHLCV 數據"""
    try:
        # 這裡改用 BybitDataFetcher 的邏輯，或者直接用 _fetch_ohlcv_data 獲取基礎K線
        exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': category,
            }
        })

        if not exchange.has['fetchOHLCV']:
            raise RuntimeError("Bybit does not support OHLCV")

        exchange.load_markets()

        since = None
        if start_date:
            since = exchange.parse8601(
                pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ')
            )

        until = None
        if end_date:
            dt = pd.to_datetime(end_date)
            if len(str(end_date)) <= 10 and dt.hour == 0 and dt.minute == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            until = exchange.parse8601(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))

        symbols_to_fetch = list(set(symbols))

        result = _fetch_ohlcv_data(
            exchange,
            symbols_to_fetch,
            timeframe,
            since,
            until,
            market_type=category
        )

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to fetch from Bybit ({category}): {e}")


def fetch_benchmark(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """獲取基準資產數據"""
    try:
        exchange = ccxt.binance()

        since = None
        if start_date:
            since = exchange.parse8601(
                pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ')
            )

        until = None
        if end_date:
            dt = pd.to_datetime(end_date)
            if len(str(end_date)) <= 10 and dt.hour == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            until = exchange.parse8601(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))

        def extract_close(df):
            return df[['timestamp', 'close']]

        factor_data = {}
        for factor in ['BTC', 'ETH']:
            df = _fetch_ohlcv_data(
                exchange,
                [factor],
                timeframe,
                since,
                until,
                extract_close
            )
            if df is not None:
                df = df.rename(columns={'close': f'{factor}_close'})
                df = df.set_index('timestamp')
                df = df[~df.index.duplicated(keep='first')]
                factor_data[factor] = df

        if not factor_data:
            warnings.warn("No factor data fetched")
            return None

        combined = pd.concat(factor_data.values(), axis=1)
        combined = combined.reset_index()

        if not symbols or combined.empty:
            return None

        dfs = []
        for sym in symbols:
            temp = combined.copy()
            temp['symbol'] = sym
            dfs.append(temp)

        return pd.concat(dfs, ignore_index=True)

    except Exception as e:
        raise RuntimeError(f"Failed to fetch benchmark: {e}")


def fetch_calendar(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """生成日曆特徵數據"""
    if not start_date or not end_date:
        return None

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        freq = TIMEFRAME_MAP.get(timeframe, 'D')
        dates = pd.date_range(start=start, end=end, freq=freq)

        if len(dates) == 0:
            return None

        is_intraday = timeframe not in ['1d', '3d', '1w', '1M']

        df_dates = pd.DataFrame({'timestamp': dates})

        df_dates['year'] = df_dates['timestamp'].dt.year
        df_dates['month'] = df_dates['timestamp'].dt.month
        df_dates['day'] = df_dates['timestamp'].dt.day
        df_dates['dayofweek'] = df_dates['timestamp'].dt.dayofweek + 1
        df_dates['dayofmonth_position'] = 1 + (df_dates['timestamp'].dt.day - 1) // 10
        df_dates['is_week_end'] = (df_dates['timestamp'].dt.dayofweek >= 5).astype(int)

        if is_intraday:
            df_dates['hour'] = df_dates['timestamp'].dt.hour
            df_dates['minute'] = df_dates['timestamp'].dt.minute
            df_dates['is_market_hours'] = 1
        else:
            df_dates['hour'] = 0
            df_dates['minute'] = 0
            df_dates['is_market_hours'] = 1

        dfs = []
        for sym in symbols:
            temp = df_dates.copy()
            temp['symbol'] = sym
            dfs.append(temp)

        return pd.concat(dfs, ignore_index=True)

    except Exception as e:
        raise RuntimeError(f"Failed to generate calendar: {e}")


def fetch_vwap(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """計算 VWAP"""
    try:
        is_long_period = timeframe in ['1d', '3d', '1w', '1M']

        if is_long_period:
            fetch_tf = '1h'
        else:
            fetch_tf = timeframe

        fetch_start = start_date
        if start_date and not is_long_period:
            fetch_start = pd.to_datetime(start_date).normalize().strftime('%Y-%m-%d %H:%M:%S')

        df = fetch_binance(symbols, fetch_tf, fetch_start, end_date)
        if df is None or df.empty:
            return None

        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['typical_price'] * df['volume']

        if is_long_period:
            if timeframe == '1d':
                df['group_key'] = df['timestamp'].dt.date
            elif timeframe == '1w':
                df['group_key'] = df['timestamp'].dt.to_period('W').dt.start_time
            elif timeframe == '1M':
                df['group_key'] = df['timestamp'].dt.to_period('M').dt.start_time
            else:
                df['group_key'] = df['timestamp'].dt.date

            agg = df.groupby(['symbol', 'group_key']).agg({
                'pv': 'sum',
                'volume': 'sum'
            }).reset_index()

            agg['vwap'] = agg['pv'] / agg['volume']
            agg = agg.rename(columns={'group_key': 'timestamp'})
            agg['timestamp'] = pd.to_datetime(agg['timestamp'])

            result_df = agg[['timestamp', 'symbol', 'vwap']]

        else:
            df['date'] = df['timestamp'].dt.date
            df['pv_cumsum'] = df.groupby(['symbol', 'date'])['pv'].cumsum()
            df['vol_cumsum'] = df.groupby(['symbol', 'date'])['volume'].cumsum()
            df['vwap'] = df['pv_cumsum'] / df['vol_cumsum']
            result_df = df[['timestamp', 'symbol', 'vwap']]

        if start_date:
            result_df = result_df[result_df['timestamp'] >= pd.to_datetime(start_date)]

        return result_df

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to calculate VWAP: {e}")


# ==============================================================================
# 數據處理與對齊
# ==============================================================================

def _process_data(
    df: pd.DataFrame,
    timeframe: str,
    user_symbols: List[str]
) -> pd.DataFrame:
    """處理數據：對齊時間範圍並填充缺失值"""
    if df.empty:
        return df

    common_start = df['timestamp'].min()
    common_end = df['timestamp'].max()

    freq = TIMEFRAME_MAP.get(timeframe, 'D')
    full_time_idx = pd.date_range(start=common_start, end=common_end, freq=freq)

    aligned_dfs = []
    grouped = df.groupby('symbol')

    for sym in user_symbols:
        if sym in grouped.groups:
            sub_df = grouped.get_group(sym).copy()
            sub_df = sub_df.drop_duplicates(subset=['timestamp'])
            sub_df = sub_df.set_index('timestamp')
            sub_df_aligned = sub_df.reindex(full_time_idx)

            price_cols = [
                c for c in sub_df_aligned.columns
                if c in ['open', 'high', 'low', 'close', 'vwap', 'BTC_close', 'ETH_close',
                         'mark_open', 'mark_high', 'mark_low', 'mark_close',
                         'index_open', 'index_high', 'index_low', 'index_close',
                         'premium_open', 'premium_high', 'premium_low', 'premium_close']
            ]
            if price_cols:
                sub_df_aligned[price_cols] = sub_df_aligned[price_cols].ffill()

            vol_cols = [
                c for c in sub_df_aligned.columns
                if 'volume' in c or 'pv' in c or 'open_interest' in c
            ]
            if vol_cols:
                sub_df_aligned[vol_cols] = sub_df_aligned[vol_cols].fillna(0)

            ratio_cols = [
                c for c in sub_df_aligned.columns
                if 'funding_rate' in c or 'ratio' in c
            ]
            if ratio_cols:
                sub_df_aligned[ratio_cols] = sub_df_aligned[ratio_cols].ffill()

            sub_df_aligned = sub_df_aligned.ffill()
            sub_df_aligned['symbol'] = sym
            sub_df_aligned.index.name = 'timestamp'
            sub_df_aligned = sub_df_aligned.reset_index()

            aligned_dfs.append(sub_df_aligned)
        else:
            warnings.warn(f"Symbol {sym} has no data after processing.")
            empty_df = pd.DataFrame(index=full_time_idx)
            empty_df.index.name = 'timestamp'
            empty_df = empty_df.reset_index()
            empty_df['symbol'] = sym
            for col in df.columns:
                if col not in ['timestamp', 'symbol']:
                    empty_df[col] = None
            aligned_dfs.append(empty_df)

    if not aligned_dfs:
        return pd.DataFrame()

    result = pd.concat(aligned_dfs, ignore_index=True)
    return result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


# ==============================================================================
# Bybit 擴展功能 (全面修正版)
# ==============================================================================

class BybitDataFetcher:
    """Bybit 數據獲取器 (修正版：修復 Interval 映射、RetCode 檢查、死循環問題與符號解析)"""

    def __init__(self, category: str = 'spot', enable_rate_limit: bool = True):
        self.category = category
        self.exchange = ccxt.bybit({
            'enableRateLimit': enable_rate_limit,
            'options': {'defaultType': category}
        })
        self.exchange.load_markets()

    def _format_symbol(self, symbol: str) -> str:
        if '/' in symbol: return symbol
        if self.category == 'linear': return f"{symbol}/USDT:USDT"
        elif self.category == 'inverse': return f"{symbol}/USD:{symbol}"
        else: return f"{symbol}/USDT"

    def _validate_symbol(self, symbol: str) -> Optional[str]:
        formatted = self._format_symbol(symbol)
        if formatted in self.exchange.symbols:
            return formatted
        # 嘗試模糊匹配
        for s in self.exchange.symbols:
            if symbol in s and 'USDT' in s:
                return s
        return None

    def _get_api_symbol(self, market_symbol: str) -> str:
        try:
            return self.exchange.market(market_symbol)['id']
        except Exception:
            return market_symbol.replace('/', '').split(':')[0]

    def _map_timeframe(self, timeframe: str) -> str:
        mapping = {
            '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
            '1d': 'D', '1w': 'W', '1M': 'M'
        }
        return mapping.get(timeframe, 'D')
    
        # 新增一個內部方法，專門負責"單一"代幣的單純獲取
    def _fetch_single_symbol_kline(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int],
        until: Optional[int],
        limit: int
    ) -> Optional[pd.DataFrame]:
        """內部方法：獲取單一代幣的 K 線數據，不處理拼接"""
        market_symbol = self._validate_symbol(symbol)
        if not market_symbol:
            return None

        try:
            # 進度提示
            print(f"    Fetching {symbol} OHLCV...", end="", flush=True)

            all_candles = []
            cursor = since
            loop_count = 0
            max_loops = 10000

            while True:
                loop_count += 1
                if loop_count > max_loops:
                    warnings.warn(f" Loop limit reached for {symbol}")
                    break
                
                if loop_count % 5 == 0:
                    print(".", end="", flush=True)

                if until and cursor and cursor > until:
                    break

                batch = self.exchange.fetch_ohlcv(
                    market_symbol,
                    timeframe,
                    since=cursor,
                    limit=limit
                )

                if not batch:
                    break

                if until:
                    batch = [c for c in batch if c[0] <= until]
                    if not batch:
                        break

                all_candles.extend(batch)

                if len(batch) < limit:
                    break

                last_ts = batch[-1][0]
                if until and last_ts >= until:
                    break

                cursor = last_ts + 1
                time.sleep(self.exchange.rateLimit / 1000)
            
            print(" Done")

            if all_candles:
                df = pd.DataFrame(
                    all_candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                return df
            return None

        except Exception as e:
            warnings.warn(f"\nFailed to fetch kline for {symbol}: {e}")
            return None

    def fetch_kline(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """獲取 OHLCV 數據 (支援自動代幣更名拼接)"""
        since_ts = None
        if start_date:
            since_ts = self.exchange.parse8601(
                pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ')
            )

        until_ts = None
        if end_date:
            dt = pd.to_datetime(end_date)
            if len(str(end_date)) <= 10 and dt.hour == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            until_ts = self.exchange.parse8601(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))

        all_dfs = []

        for symbol in symbols:
            # 1. 先嘗試獲取當前符號的數據 (例如 POL)
            df_main = self._fetch_single_symbol_kline(symbol, timeframe, since_ts, until_ts, limit)
            
            # 檢查是否需要拼接舊代幣
            rename_info = SYMBOL_RENAMES.get(symbol)
            
            if rename_info:
                old_symbol = rename_info['old_symbol']
                
                # 確定主數據的最早時間
                main_start_ts = None
                if df_main is not None and not df_main.empty:
                    main_start_ts = df_main['timestamp'].min().timestamp() * 1000
                
                # 如果主數據是空的，或者主數據的開始時間晚於我們請求的開始時間
                # 則需要去抓舊代幣
                need_old_data = False
                old_until_ts = until_ts

                if df_main is None or df_main.empty:
                    # 完全沒抓到新代幣，嘗試全程抓舊代幣
                    need_old_data = True
                elif since_ts is not None and main_start_ts > since_ts + (1000 * 60): # 容許1分鐘誤差
                    # 新代幣數據不夠長，前面缺一塊
                    need_old_data = True
                    # 舊數據只需要抓到新數據開始之前
                    old_until_ts = main_start_ts - 1
                
                if need_old_data:
                    print(f"    Detected gap for {symbol}, attempting to fetch old symbol: {old_symbol}")
                    df_old = self._fetch_single_symbol_kline(old_symbol, timeframe, since_ts, old_until_ts, limit)
                    
                    if df_old is not None and not df_old.empty:
                        # 將舊代幣名稱重命名為新代幣
                        df_old['symbol'] = symbol
                        
                        if df_main is not None:
                            df_main = pd.concat([df_old, df_main], ignore_index=True)
                        else:
                            df_main = df_old

            if df_main is not None and not df_main.empty:
                # 去重並排序
                df_main = df_main.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
                all_dfs.append(df_main)

        if not all_dfs:
            return None

        return pd.concat(all_dfs, ignore_index=True)

    def fetch_kline(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """獲取 OHLCV 數據 (支援自動代幣更名拼接)"""
        since_ts = None
        if start_date:
            since_ts = self.exchange.parse8601(
                pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ')
            )

        until_ts = None
        if end_date:
            dt = pd.to_datetime(end_date)
            if len(str(end_date)) <= 10 and dt.hour == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            until_ts = self.exchange.parse8601(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))

        all_dfs = []

        for symbol in symbols:
            # 1. 先嘗試獲取當前符號的數據 (例如 POL)
            df_main = self._fetch_single_symbol_kline(symbol, timeframe, since_ts, until_ts, limit)
            
            # 檢查是否需要拼接舊代幣
            rename_info = SYMBOL_RENAMES.get(symbol)
            
            if rename_info:
                old_symbol = rename_info['old_symbol']
                
                # 確定主數據的最早時間
                main_start_ts = None
                if df_main is not None and not df_main.empty:
                    main_start_ts = df_main['timestamp'].min().timestamp() * 1000
                
                # 如果主數據是空的，或者主數據的開始時間晚於我們請求的開始時間
                # 則需要去抓舊代幣
                need_old_data = False
                old_until_ts = until_ts

                if df_main is None or df_main.empty:
                    # 完全沒抓到新代幣，嘗試全程抓舊代幣
                    need_old_data = True
                elif since_ts is not None and main_start_ts > since_ts + (1000 * 60): # 容許1分鐘誤差
                    # 新代幣數據不夠長，前面缺一塊
                    need_old_data = True
                    # 舊數據只需要抓到新數據開始之前
                    old_until_ts = main_start_ts - 1
                
                if need_old_data:
                    print(f"    Detected gap for {symbol}, attempting to fetch old symbol: {old_symbol}")
                    df_old = self._fetch_single_symbol_kline(old_symbol, timeframe, since_ts, old_until_ts, limit)
                    
                    if df_old is not None and not df_old.empty:
                        # 將舊代幣名稱重命名為新代幣
                        df_old['symbol'] = symbol
                        
                        if df_main is not None:
                            df_main = pd.concat([df_old, df_main], ignore_index=True)
                        else:
                            df_main = df_old

            if df_main is not None and not df_main.empty:
                # 去重並排序
                df_main = df_main.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
                all_dfs.append(df_main)

        if not all_dfs:
            return None

        return pd.concat(all_dfs, ignore_index=True)
    
    def _fetch_price_kline(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str],
        api_method: str,
        limit: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        通用價格 K線獲取方法 (含自動拼接邏輯)
        此方法也進行修改以支持 Mark Price / Index Price 的拼接
        """
        since_ts = None
        if start_date:
            since_ts = self.exchange.parse8601(
                pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ')
            )

        until_ts = None
        if end_date:
            dt = pd.to_datetime(end_date)
            if len(str(end_date)) <= 10 and dt.hour == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            until_ts = self.exchange.parse8601(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))

        bybit_interval = self._map_timeframe(timeframe)

        # 定義內部的單次獲取邏輯
        def _fetch_single_raw(target_sym, start_ts, end_ts):
            market_symbol = self._validate_symbol(target_sym)
            if not market_symbol: return None
            
            try:
                print(f"    Fetching {target_sym} ({api_method})...", end="", flush=True)
                api_symbol = self._get_api_symbol(market_symbol)
                candles = []
                cursor = start_ts
                loop_count = 0
                max_loops = 500
                last_batch_max_ts = 0

                while True:
                    loop_count += 1
                    if loop_count > max_loops: break
                    if loop_count % 5 == 0: print(".", end="", flush=True)
                    if end_ts and cursor and cursor > end_ts: break

                    params = {
                        'category': self.category,
                        'symbol': api_symbol,
                        'interval': bybit_interval,
                        'limit': limit
                    }
                    if cursor: params['start'] = cursor
                    if end_ts: params['end'] = end_ts

                    try:
                        method = getattr(self.exchange, api_method)
                        response = method(params)
                    except Exception as e:
                        warnings.warn(f" API fail: {e}")
                        break

                    if int(response['retCode']) != 0: break # Error

                    batch = response['result']['list']
                    if not batch: break

                    temp_batch = []
                    for candle in batch:
                        temp_batch.append({
                            'timestamp': int(candle[0]),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'symbol': target_sym # 使用傳入的 target_sym
                        })
                    
                    temp_batch.sort(key=lambda x: x['timestamp'])
                    if end_ts:
                        temp_batch = [c for c in temp_batch if c['timestamp'] <= end_ts]
                    if not temp_batch: break
                    
                    current_max = temp_batch[-1]['timestamp']
                    if current_max <= last_batch_max_ts and loop_count > 1: break
                    last_batch_max_ts = current_max

                    candles.extend(temp_batch)
                    if len(batch) < limit: break
                    
                    last_ts = temp_batch[-1]['timestamp']
                    if end_ts and last_ts >= end_ts: break
                    cursor = last_ts + 1
                    time.sleep(self.exchange.rateLimit / 1000)
                
                print(" Done")
                if candles:
                    return pd.DataFrame(candles)
                return None
            except Exception as e:
                warnings.warn(f" Error: {e}")
                return None

        all_dfs = []

        for symbol in symbols:
            # 1. 抓取新代幣
            df_main = _fetch_single_raw(symbol, since_ts, until_ts)
            
            # 2. 處理拼接
            rename_info = SYMBOL_RENAMES.get(symbol)
            if rename_info:
                old_symbol = rename_info['old_symbol']
                main_start_ts = None
                if df_main is not None and not df_main.empty:
                    main_start_ts = df_main['timestamp'].iloc[0] # int ms
                
                need_old = False
                old_until = until_ts
                
                if df_main is None or df_main.empty:
                    need_old = True
                elif since_ts is not None and main_start_ts > since_ts + 60000:
                    need_old = True
                    old_until = main_start_ts - 1
                
                if need_old:
                    print(f"    [Stitch] Fetching old symbol {old_symbol} for {symbol}...")
                    df_old = _fetch_single_raw(old_symbol, since_ts, old_until)
                    if df_old is not None and not df_old.empty:
                        df_old['symbol'] = symbol # 改名
                        if df_main is not None:
                            df_main = pd.concat([df_old, df_main], ignore_index=True)
                        else:
                            df_main = df_old

            if df_main is not None and not df_main.empty:
                df_main['timestamp'] = pd.to_datetime(df_main['timestamp'], unit='ms')
                df_main = df_main.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
                all_dfs.append(df_main)

        if not all_dfs:
            return None

        return pd.concat(all_dfs, ignore_index=True)
    
    def fetch_mark_price_kline(self, symbols, timeframe='1d', start_date=None, end_date=None, limit=200):
        if self.category not in ['linear', 'inverse']:
            return None
        return self._fetch_price_kline(symbols, timeframe, start_date, end_date,
                                     'publicGetV5MarketMarkPriceKline', limit)

    def fetch_index_price_kline(self, symbols, timeframe='1d', start_date=None, end_date=None, limit=200):
        if self.category not in ['linear', 'inverse']:
            return None
        return self._fetch_price_kline(symbols, timeframe, start_date, end_date,
                                     'publicGetV5MarketIndexPriceKline', limit)

    def fetch_premium_index_kline(self, symbols, timeframe='1d', start_date=None, end_date=None, limit=200):
        if self.category != 'linear':
            return None
        return self._fetch_price_kline(symbols, timeframe, start_date, end_date,
                                     'publicGetV5MarketPremiumIndexPriceKline', limit)

    def fetch_funding_rate(self, symbols, start_date=None, end_date=None, limit=200):
        """獲取資金費率"""
        if self.category not in ['linear', 'inverse']:
            return None

        since = None
        if start_date:
            since = self.exchange.parse8601(pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ'))

        until = None
        if end_date:
            dt = pd.to_datetime(end_date)
            if len(str(end_date)) <= 10 and dt.hour == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            until = self.exchange.parse8601(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))

        all_dfs = []

        for symbol in symbols:
            market_symbol = self._validate_symbol(symbol)
            if not market_symbol:
                continue

            try:
                print(f"    Fetching {symbol} Funding...", end="", flush=True)
                all_rates = []
                cursor = since

                loop_count = 0
                max_loops = 500
                last_batch_max_ts = 0

                while True:
                    loop_count += 1
                    if loop_count > max_loops:
                        warnings.warn(f" Loop limit reached for {symbol}")
                        break
                    
                    if loop_count % 5 == 0:
                        print(".", end="", flush=True)

                    if until and cursor and cursor > until:
                        break

                    try:
                        batch = self.exchange.fetch_funding_rate_history(
                            market_symbol, since=cursor, limit=limit
                        )
                    except Exception:
                        api_symbol = self._get_api_symbol(market_symbol)
                        api_params = {
                            'category': self.category,
                            'symbol': api_symbol,
                            'limit': limit
                        }
                        if cursor:
                            api_params['startTime'] = cursor
                        if until:
                            api_params['endTime'] = until

                        response = self.exchange.publicGetV5MarketFundingHistory(api_params)
                        
                        if int(response['retCode']) != 0:
                            break
                        
                        batch = []
                        for item in response['result']['list']:
                            batch.append({
                                'timestamp': int(item['fundingRateTimestamp']),
                                'fundingRate': float(item['fundingRate']),
                                'symbol': symbol
                            })

                    if not batch:
                        break

                    processed_batch = []
                    is_dict = isinstance(batch[0], dict)
                    
                    batch.sort(key=lambda x: x.get('timestamp') or x.get('fundingRateTimestamp') if is_dict else x['timestamp'])

                    for rate in batch:
                        ts = int(rate.get('timestamp', 0) or rate.get('fundingRateTimestamp', 0)) if is_dict else int(rate['timestamp'])
                        fr = float(rate.get('fundingRate', 0)) if is_dict else float(rate['fundingRate'])
                        
                        if until and ts > until:
                            continue
                        
                        processed_batch.append({
                            'timestamp': ts,
                            'funding_rate': fr,
                            'symbol': symbol
                        })
                    
                    if not processed_batch:
                        break
                    
                    current_batch_max_ts = processed_batch[-1]['timestamp']
                    if current_batch_max_ts <= last_batch_max_ts and loop_count > 1:
                        break
                    last_batch_max_ts = current_batch_max_ts

                    all_rates.extend(processed_batch)

                    if len(batch) < limit:
                        break

                    last_ts = processed_batch[-1]['timestamp']
                    if until and last_ts >= until:
                        break

                    cursor = last_ts + 1
                    time.sleep(self.exchange.rateLimit / 1000)
                
                print(" Done")

                if all_rates:
                    df = pd.DataFrame(all_rates)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    all_dfs.append(df)

            except Exception as e:
                warnings.warn(f"\nFailed to fetch funding rate for {symbol}: {e}")
                continue

        if not all_dfs:
            return None

        return pd.concat(all_dfs, ignore_index=True)

    def fetch_open_interest(self, symbols, timeframe='1h', start_date=None, end_date=None, limit=200):
        """獲取持倉量"""
        if self.category not in ['linear', 'inverse']:
            return None

        interval_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1d'}
        if timeframe not in interval_map:
            timeframe = '1h'
        api_interval = interval_map[timeframe]

        since = None
        if start_date:
            since = self.exchange.parse8601(pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ'))

        until = None
        if end_date:
            dt = pd.to_datetime(end_date)
            if len(str(end_date)) <= 10 and dt.hour == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            until = self.exchange.parse8601(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))

        all_dfs = []

        for symbol in symbols:
            market_symbol = self._validate_symbol(symbol)
            if not market_symbol:
                continue

            try:
                print(f"    Fetching {symbol} OI...", end="", flush=True)
                api_symbol = self._get_api_symbol(market_symbol)
                
                all_oi = []
                cursor = since

                loop_count = 0
                max_loops = 500
                last_batch_max_ts = 0

                while True:
                    loop_count += 1
                    if loop_count > max_loops:
                        warnings.warn(f" Loop limit reached for {symbol}")
                        break
                    
                    if loop_count % 5 == 0:
                        print(".", end="", flush=True)

                    if until and cursor and cursor > until:
                        break

                    params = {
                        'category': self.category,
                        'symbol': api_symbol,
                        'intervalTime': api_interval,
                        'limit': limit
                    }
                    if cursor:
                        params['startTime'] = cursor
                    if until:
                        params['endTime'] = until

                    try:
                        response = self.exchange.publicGetV5MarketOpenInterest(params)
                    except AttributeError:
                        break

                    if int(response['retCode']) != 0:
                        warnings.warn(f"\nBybit API Error for {symbol}: {response['retMsg']}")
                        break

                    batch = response['result']['list']
                    if not batch:
                        break

                    temp_batch = []
                    for oi in batch:
                        temp_batch.append({
                            'timestamp': int(oi['timestamp']),
                            'open_interest': float(oi['openInterest']),
                            'symbol': symbol
                        })
                    
                    temp_batch.sort(key=lambda x: x['timestamp'])

                    if until:
                        temp_batch = [x for x in temp_batch if x['timestamp'] <= until]
                    
                    if not temp_batch:
                        break

                    current_batch_max_ts = temp_batch[-1]['timestamp']
                    if current_batch_max_ts <= last_batch_max_ts and loop_count > 1:
                        break
                    last_batch_max_ts = current_batch_max_ts

                    all_oi.extend(temp_batch)

                    if len(batch) < limit:
                        break

                    last_ts = temp_batch[-1]['timestamp']
                    if until and last_ts >= until:
                        break

                    cursor = last_ts + 1
                    time.sleep(self.exchange.rateLimit / 1000)
                
                print(" Done")

                if all_oi:
                    df = pd.DataFrame(all_oi)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    all_dfs.append(df)

            except Exception as e:
                warnings.warn(f"\nFailed to fetch open interest for {symbol}: {e}")
                continue

        if not all_dfs:
            return None

        return pd.concat(all_dfs, ignore_index=True)

    def fetch_long_short_ratio(self, symbols, timeframe='1h', start_date=None, end_date=None, limit=200):
        """獲取多空比"""
        if self.category not in ['linear', 'inverse']:
            return None

        since = None
        if start_date:
            since = self.exchange.parse8601(pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ'))

        until = None
        if end_date:
            dt = pd.to_datetime(end_date)
            if len(str(end_date)) <= 10 and dt.hour == 0:
                dt = dt.replace(hour=23, minute=59, second=59)
            until = self.exchange.parse8601(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))

        all_dfs = []

        for symbol in symbols:
            market_symbol = self._validate_symbol(symbol)
            if not market_symbol:
                continue

            try:
                print(f"    Fetching {symbol} LS Ratio...", end="", flush=True)
                api_symbol = self._get_api_symbol(market_symbol)

                all_ratio = []
                cursor = since

                loop_count = 0
                max_loops = 500
                last_batch_max_ts = 0

                while True:
                    loop_count += 1
                    if loop_count > max_loops:
                        warnings.warn(f" Loop limit reached for {symbol}")
                        break
                    
                    if loop_count % 5 == 0:
                        print(".", end="", flush=True)

                    if until and cursor and cursor > until:
                        break

                    params = {
                        'category': self.category,
                        'symbol': api_symbol,
                        'period': timeframe,
                        'limit': limit
                    }
                    if cursor:
                        params['startTime'] = cursor
                    if until:
                        params['endTime'] = until

                    try:
                        response = self.exchange.publicGetV5MarketAccountRatio(params)
                    except AttributeError:
                        break

                    if int(response['retCode']) != 0:
                        break

                    batch = response['result']['list']
                    if not batch:
                        break

                    temp_batch = []
                    for ratio in batch:
                        temp_batch.append({
                            'timestamp': int(ratio['timestamp']),
                            'long_account_ratio': float(ratio['buyRatio']),
                            'short_account_ratio': float(ratio['sellRatio']),
                            'symbol': symbol
                        })
                    
                    temp_batch.sort(key=lambda x: x['timestamp'])

                    if until:
                        temp_batch = [x for x in temp_batch if x['timestamp'] <= until]
                    
                    if not temp_batch:
                        break
                    
                    current_batch_max_ts = temp_batch[-1]['timestamp']
                    if current_batch_max_ts <= last_batch_max_ts and loop_count > 1:
                        break
                    last_batch_max_ts = current_batch_max_ts

                    all_ratio.extend(temp_batch)

                    if len(batch) < limit:
                        break

                    last_ts = temp_batch[-1]['timestamp']
                    if until and last_ts >= until:
                        break

                    cursor = last_ts + 1
                    time.sleep(self.exchange.rateLimit / 1000)
                
                print(" Done")

                if all_ratio:
                    df = pd.DataFrame(all_ratio)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    all_dfs.append(df)

            except Exception as e:
                warnings.warn(f"\nFailed to fetch long/short ratio for {symbol}: {e}")
                continue

        if not all_dfs:
            return None

        return pd.concat(all_dfs, ignore_index=True)

    def fetch_tickers(self, symbols):
        """獲取行情快照"""
        all_tickers = []
        for symbol in symbols:
            market_symbol = self._validate_symbol(symbol)
            if not market_symbol:
                continue
            try:
                ticker = self.exchange.fetch_ticker(market_symbol)
                ticker_data = {
                    'timestamp': pd.to_datetime(ticker['timestamp'], unit='ms'),
                    'symbol': symbol,
                    'last_price': ticker['last'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'high_24h': ticker['high'],
                    'low_24h': ticker['low'],
                    'volume_24h': ticker['baseVolume'],
                    'quote_volume_24h': ticker['quoteVolume'],
                    'price_change_24h': ticker['change'],
                    'price_change_percent_24h': ticker['percentage'],
                }
                all_tickers.append(ticker_data)
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                warnings.warn(f"Failed to fetch ticker for {symbol}: {e}")
                continue

        if not all_tickers:
            return None
        return pd.DataFrame(all_tickers)

    def fetch_orderbook(self, symbols, depth=50):
        """獲取訂單簿"""
        orderbooks = {}
        for symbol in symbols:
            market_symbol = self._validate_symbol(symbol)
            if not market_symbol:
                continue
            try:
                orderbook = self.exchange.fetch_order_book(market_symbol, limit=depth)
                bids_df = pd.DataFrame(orderbook['bids'], columns=['price', 'amount'])
                bids_df['side'] = 'bid'
                asks_df = pd.DataFrame(orderbook['asks'], columns=['price', 'amount'])
                asks_df['side'] = 'ask'
                combined = pd.concat([bids_df, asks_df], ignore_index=True)
                combined['timestamp'] = pd.to_datetime(orderbook['timestamp'], unit='ms')
                combined['symbol'] = symbol
                orderbooks[symbol] = combined
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                warnings.warn(f"Failed to fetch orderbook for {symbol}: {e}")
                continue

        if not orderbooks:
            return None
        return orderbooks

    def fetch_recent_trades(self, symbols, limit=1000):
        """獲取最近交易"""
        all_trades = []
        for symbol in symbols:
            market_symbol = self._validate_symbol(symbol)
            if not market_symbol:
                continue
            try:
                trades = self.exchange.fetch_trades(market_symbol, limit=limit)
                for trade in trades:
                    all_trades.append({
                        'timestamp': pd.to_datetime(trade['timestamp'], unit='ms'),
                        'symbol': symbol,
                        'trade_id': trade['id'],
                        'price': trade['price'],
                        'amount': trade['amount'],
                        'side': trade['side'],
                        'cost': trade['cost']
                    })
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                warnings.warn(f"Failed to fetch trades for {symbol}: {e}")
                continue

        if not all_trades:
            return None
        return pd.DataFrame(all_trades)

    def fetch_all_data(self, symbols, timeframe='1d', start_date=None, end_date=None, include_derivatives_data=True):
        """一次性獲取所有數據"""
        result = {}

        print(f"Fetching OHLCV data...")
        kline = self.fetch_kline(symbols, timeframe, start_date, end_date)
        if kline is not None:
            result['kline'] = kline
            print(f"  ✓ OHLCV: {len(kline)} rows")
        else:
            print(f"  ✗ OHLCV: Failed")

        if include_derivatives_data and self.category in ['linear', 'inverse']:
            print(f"Fetching mark price kline...")
            mark_kline = self.fetch_mark_price_kline(symbols, timeframe, start_date, end_date)
            if mark_kline is not None:
                result['mark_kline'] = mark_kline
                print(f"  ✓ Mark price: {len(mark_kline)} rows")
            else:
                print(f"  ✗ Mark price: Failed")

            print(f"Fetching index price kline...")
            index_kline = self.fetch_index_price_kline(symbols, timeframe, start_date, end_date)
            if index_kline is not None:
                result['index_kline'] = index_kline
                print(f"  ✓ Index price: {len(index_kline)} rows")
            else:
                print(f"  ✗ Index price: Failed")

            if self.category == 'linear':
                print(f"Fetching premium index kline...")
                premium_kline = self.fetch_premium_index_kline(symbols, timeframe, start_date, end_date)
                if premium_kline is not None:
                    result['premium_kline'] = premium_kline
                    print(f"  ✓ Premium index: {len(premium_kline)} rows")
                else:
                    print(f"  ✗ Premium index: Failed")

            print(f"Fetching funding rate...")
            funding_rate = self.fetch_funding_rate(symbols, start_date, end_date)
            if funding_rate is not None:
                result['funding_rate'] = funding_rate
                print(f"  ✓ Funding rate: {len(funding_rate)} rows")
            else:
                print(f"  ✗ Funding rate: Failed")

            print(f"Fetching open interest...")
            open_interest = self.fetch_open_interest(symbols, timeframe, start_date, end_date)
            if open_interest is not None:
                result['open_interest'] = open_interest
                print(f"  ✓ Open interest: {len(open_interest)} rows")
            else:
                print(f"  ✗ Open interest: Failed")

            print(f"Fetching long/short ratio...")
            long_short_ratio = self.fetch_long_short_ratio(symbols, timeframe, start_date, end_date)
            if long_short_ratio is not None:
                result['long_short_ratio'] = long_short_ratio
                print(f"  ✓ Long/short ratio: {len(long_short_ratio)} rows")
            else:
                print(f"  ✗ Long/short ratio: Failed")

        print(f"Fetching tickers...")
        tickers = self.fetch_tickers(symbols)
        if tickers is not None:
            result['tickers'] = tickers
            print(f"  ✓ Tickers: {len(tickers)} rows")
        else:
            print(f"  ✗ Tickers: Failed")

        print(f"Fetching orderbook...")
        orderbook = self.fetch_orderbook(symbols, depth=50)
        if orderbook is not None:
            result['orderbook'] = orderbook
            print(f"  ✓ Orderbook: {len(orderbook)} symbols")
        else:
            print(f"  ✗ Orderbook: Failed")

        print(f"Fetching recent trades...")
        trades = self.fetch_recent_trades(symbols, limit=500)
        if trades is not None:
            result['recent_trades'] = trades
            print(f"  ✓ Recent trades: {len(trades)} rows")
        else:
            print(f"  ✗ Recent trades: Failed")

        return result


def fetch_bybit_all(
    symbols: List[str],
    timeframe: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: str = 'spot',
    include_derivatives_data: bool = True,
    output_dir: Optional[str] = None,
    return_panel: bool = True
) -> Union['Panel', Dict[str, pd.DataFrame]]:
    """一次性獲取 Bybit 所有數據"""
    fetcher = BybitDataFetcher(category=category)
    result = fetcher.fetch_all_data(symbols, timeframe, start_date, end_date, include_derivatives_data)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for data_type, df in result.items():
            if isinstance(df, pd.DataFrame):
                output_path = os.path.join(output_dir, f'{data_type}_{category}.csv')
                df.to_csv(output_path, index=False)
            elif isinstance(df, dict):
                for sym, ob_df in df.items():
                    safe_sym = sym.replace('/', '_').replace(':', '_')
                    output_path = os.path.join(output_dir, f'{data_type}_{safe_sym}_{category}.csv')
                    ob_df.to_csv(output_path, index=False)

    if return_panel:
        if not result:
            raise ValueError("No data fetched")

        base_df = result.get('kline')
        if base_df is None:
            raise ValueError("No kline data available")

        print(f"\nMerging data...")
        snapshots = {}

        kline_types_with_rename = {
            'mark_kline': 'mark',
            'index_kline': 'index',
            'premium_kline': 'premium'
        }

        timeseries_types = ['funding_rate', 'open_interest', 'long_short_ratio']
        snapshot_types = ['tickers', 'recent_trades', 'orderbook']

        for data_type, prefix in kline_types_with_rename.items():
            if data_type not in result:
                continue
            df = result[data_type]
            if df is None or df.empty:
                continue

            rename_map = {
                'open': f'{prefix}_open',
                'high': f'{prefix}_high',
                'low': f'{prefix}_low',
                'close': f'{prefix}_close'
            }
            # 移除重複的列，避免 merge 衝突
            cols_to_use = ['timestamp', 'symbol'] + [c for c in df.columns if c in rename_map]
            df_subset = df[cols_to_use].rename(columns=rename_map)
            
            base_df = pd.merge(base_df, df_subset, on=['timestamp', 'symbol'], how='left')

        for data_type in timeseries_types:
            if data_type not in result:
                continue
            df = result[data_type]
            if df is None or df.empty:
                continue
            # 這裡簡單merge，實際應用可能需要對 timestamp 做 asof merge 或重採樣
            base_df = pd.merge(base_df, df, on=['timestamp', 'symbol'], how='left')

        for snapshot_type in snapshot_types:
            if snapshot_type in result and result[snapshot_type] is not None:
                snapshots[snapshot_type] = result[snapshot_type]

        try:
            from .panel import Panel
            panel_data = {'kline': base_df, **snapshots}
            return Panel(panel_data)
        except ImportError:
            return base_df
    else:
        return result


def calculate_orderbook_metrics(orderbook_df):
    """計算訂單簿指標"""
    bids = orderbook_df[orderbook_df['side'] == 'bid'].copy()
    asks = orderbook_df[orderbook_df['side'] == 'ask'].copy()

    total_bid_volume = bids['amount'].sum()
    total_ask_volume = asks['amount'].sum()
    total_bid_value = (bids['price'] * bids['amount']).sum()
    total_ask_value = (asks['price'] * asks['amount']).sum()

    volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    value_imbalance = (total_bid_value - total_ask_value) / (total_bid_value + total_ask_value)

    best_bid = bids['price'].max()
    best_ask = asks['price'].min()
    spread = best_ask - best_bid
    spread_pct = (spread / best_bid) * 100
    mid_price = (best_bid + best_ask) / 2

    bid_threshold = mid_price * 0.995
    ask_threshold = mid_price * 1.005
    bid_depth_05pct = bids[bids['price'] >= bid_threshold]['amount'].sum()
    ask_depth_05pct = asks[asks['price'] <= ask_threshold]['amount'].sum()

    return {
        'timestamp': orderbook_df['timestamp'].iloc[0],
        'symbol': orderbook_df['symbol'].iloc[0],
        'best_bid': best_bid,
        'best_ask': best_ask,
        'mid_price': mid_price,
        'spread': spread,
        'spread_pct': spread_pct,
        'total_bid_volume': total_bid_volume,
        'total_ask_volume': total_ask_volume,
        'volume_imbalance': volume_imbalance,
        'value_imbalance': value_imbalance,
        'bid_depth_05pct': bid_depth_05pct,
        'ask_depth_05pct': ask_depth_05pct,
        'bid_ask_depth_ratio': bid_depth_05pct / ask_depth_05pct if ask_depth_05pct > 0 else 0,
    }
