"""Multi-column market data container with flat (timestamp, symbol) structure."""

import pandas as pd
from typing import Union, Optional, List, Dict, Any
from .core import Factor

class Panel:
    """Multi-column market data container.
    
    Stores OHLCV and derived data in a flat DataFrame with
    columns ['timestamp', 'symbol', ...].
    
    Can also store snapshot data (like orderbook) separately.
    """

    def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        """Initialize Panel with time-series data and optional snapshot data."""
        
        # 定義需要強制位移（Shift）以避免前瞻偏差的敏感欄位
        # [優化點 4] 這些數據通常是 K 線結束或結算時才產生的，回測時必須延後一根 K 線使用
        LAG_SENSITIVE_KEYS = ['funding_rate', 'open_interest', 'long_short_ratio']

        # 處理字典輸入（來自 fetch_bybit_all）
        if isinstance(data, dict):
            # 提取時間序列數據
            if 'timeseries' in data:
                df = data['timeseries'].copy()
            elif 'kline' in data:
                # 如果直接傳入 fetch_bybit_all 的原始結果
                df = data['kline'].copy()
                
                # 合併其他時間序列數據
                # 定義一般不需要位移的欄位 (例如 Mark Price, Index Price 本身就是價格序列)
                normal_keys = ['mark_kline', 'index_kline', 'premium_kline']
                
                # 1. 處理普通數據 (直接 Merge)
                for key in normal_keys:
                    if key in data and data[key] is not None:
                        df = pd.merge(df, data[key], on=['timestamp', 'symbol'], how='left')

                # 2. [關鍵修改] 處理敏感數據 (Shift 後再 Merge)
                for key in LAG_SENSITIVE_KEYS:
                    if key in data and data[key] is not None:
                        temp_df = data[key].copy()
                        
                        # 確保排序正確
                        temp_df = temp_df.sort_values(['symbol', 'timestamp'])
                        
                        # 找出數值欄位 (排除 timestamp, symbol)
                        val_cols = [c for c in temp_df.columns if c not in ['timestamp', 'symbol']]
                        
                        if val_cols:
                            # 對每個 symbol 分組後，將數值向後位移一格
                            # 這樣 T 時刻看到的，就是 T-1 時刻產生的數據（安全！）
                            temp_df[val_cols] = temp_df.groupby('symbol')[val_cols].shift(1)
                            
                            print(f"Safety: Applied 1-period lag to '{key}' to prevent look-ahead bias.")
                        
                        df = pd.merge(df, temp_df, on=['timestamp', 'symbol'], how='left')
            else:
                raise ValueError("Dict input must contain 'timeseries' or 'kline' key")
            
            # 存儲快照數據 (這部分不變)
            self.snapshots = {}
            for key in ['orderbook', 'tickers', 'recent_trades']:
                if key in data and data[key] is not None:
                    self.snapshots[key] = data[key]
        else:
            # 傳統 DataFrame 輸入
            df = data.copy()
            self.snapshots = {}
        
        # 處理 MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        
        # 驗證必要欄位
        if 'timestamp' not in df.columns or 'symbol' not in df.columns:
            raise ValueError("Data must have 'timestamp' and 'symbol' columns")
        
        # 標準化時間戳並排序
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        # 移除重複列（保留第一次出現的，防止 merge 產生的 duplicates）
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        self.data = df
    @classmethod
    def from_csv(cls, path: str) -> 'Panel':
        """從 CSV 文件加載 Panel"""
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return cls(df)
    
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> 'Panel':
        """從 DataFrame 創建 Panel"""
        return cls(df)
    
    @classmethod
    def from_bybit_all(cls, bybit_data: Dict[str, Any]) -> 'Panel':
        """從 fetch_bybit_all 的返回結果創建 Panel
        
        Args:
            bybit_data: fetch_bybit_all 返回的字典
            
        Returns:
            Panel 對象
        """
        return cls(bybit_data)
    
    def to_df(self) -> pd.DataFrame:
        """返回時間序列數據的 DataFrame"""
        return self.data.copy()
    
    def get_orderbook(self, symbol: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """獲取訂單簿數據
        
        Args:
            symbol: 如果指定，返回該交易對的訂單簿；否則返回所有訂單簿
            
        Returns:
            單個訂單簿 DataFrame 或訂單簿字典
        """
        if 'orderbook' not in self.snapshots:
            raise ValueError("No orderbook data available")
        
        orderbooks = self.snapshots['orderbook']
        
        if symbol is not None:
            if symbol not in orderbooks:
                raise ValueError(f"Orderbook for {symbol} not found")
            return orderbooks[symbol].copy()
        
        return orderbooks
    
    def get_tickers(self) -> pd.DataFrame:
        """獲取行情快照數據"""
        if 'tickers' not in self.snapshots:
            raise ValueError("No tickers data available")
        return self.snapshots['tickers'].copy()
    
    def get_recent_trades(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """獲取最近交易數據
        
        Args:
            symbol: 如果指定，只返回該交易對的交易數據
        """
        if 'recent_trades' not in self.snapshots:
            raise ValueError("No recent trades data available")
        
        trades = self.snapshots['recent_trades'].copy()
        
        if symbol is not None:
            trades = trades[trades['symbol'] == symbol]
        
        return trades
    
    def has_snapshot(self, snapshot_type: str) -> bool:
        """檢查是否有特定類型的快照數據
        
        Args:
            snapshot_type: 'orderbook', 'tickers', 'recent_trades' 等
        """
        return snapshot_type in self.snapshots
    
    def list_snapshots(self) -> List[str]:
        """列出所有可用的快照數據類型"""
        return list(self.snapshots.keys())
    
    def __getitem__(self, key) -> Union[Factor, 'Panel']:
        """獲取列或多列數據
        
        支持：
        - panel['close']: 返回 Factor
        - panel[['close', 'volume']]: 返回 Panel
        - panel['orderbook']: 返回訂單簿數據（如果存在）
        """
        # 檢查是否為快照數據
        if isinstance(key, str) and key in self.snapshots:
            return self.snapshots[key]
        
        # 處理時間序列數據
        if isinstance(key, str):
            if key not in self.data.columns:
                raise ValueError(f"Column '{key}' not found. Available columns: {self.columns}")
            factor_data = self.data[['timestamp', 'symbol', key]].copy()
            factor_data.columns = ['timestamp', 'symbol', 'factor']
            return Factor(factor_data, key)
        elif isinstance(key, list):
            # 檢查所有列是否存在
            missing_cols = [c for c in key if c not in self.data.columns and c not in ['timestamp', 'symbol']]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}. Available: {self.columns}")
            
            cols = ['timestamp', 'symbol'] + [c for c in key if c not in ['timestamp', 'symbol']]
            return Panel(self.data[cols].copy())
        else:
            raise TypeError("Key must be str or list")
    
    def slice_time(self, start: Optional[str] = None, end: Optional[str] = None) -> 'Panel':
        """按時間範圍切片"""
        mask = pd.Series(True, index=self.data.index)
        if start:
            mask &= self.data['timestamp'] >= pd.to_datetime(start)
        if end:
            mask &= self.data['timestamp'] <= pd.to_datetime(end)
        
        # 創建新的 Panel，保留快照數據
        new_panel = Panel(self.data[mask].copy())
        new_panel.snapshots = self.snapshots.copy()
        return new_panel
    
    def slice_symbols(self, symbols: Union[str, List[str]]) -> 'Panel':
        """按交易對切片"""
        if isinstance(symbols, str):
            symbols = [symbols]
        mask = self.data['symbol'].isin(symbols)
        
        # 創建新的 Panel
        new_panel = Panel(self.data[mask].copy())
        
        # 過濾快照數據
        if 'orderbook' in self.snapshots:
            new_panel.snapshots['orderbook'] = {
                sym: ob for sym, ob in self.snapshots['orderbook'].items() 
                if sym in symbols
            }
        
        if 'tickers' in self.snapshots:
            tickers = self.snapshots['tickers']
            new_panel.snapshots['tickers'] = tickers[tickers['symbol'].isin(symbols)]
        
        if 'recent_trades' in self.snapshots:
            trades = self.snapshots['recent_trades']
            new_panel.snapshots['recent_trades'] = trades[trades['symbol'].isin(symbols)]
        
        return new_panel
    
    def to_csv(self, path: str, save_snapshots: bool = False) -> str:
        """保存為 CSV
        
        Args:
            path: 時間序列數據的保存路徑
            save_snapshots: 是否同時保存快照數據
        """
        import os
        
        # 保存時間序列數據
        self.data.to_csv(path, index=False)
        
        # 保存快照數據
        if save_snapshots and self.snapshots:
            base_dir = os.path.dirname(path)
            base_name = os.path.splitext(os.path.basename(path))[0]
            
            for snapshot_type, snapshot_data in self.snapshots.items():
                if snapshot_type == 'orderbook':
                    # 訂單簿是字典，每個交易對單獨保存
                    for symbol, ob_df in snapshot_data.items():
                        ob_path = os.path.join(base_dir, f"{base_name}_orderbook_{symbol}.csv")
                        ob_df.to_csv(ob_path, index=False)
                elif isinstance(snapshot_data, pd.DataFrame):
                    # 其他快照數據直接保存
                    snapshot_path = os.path.join(base_dir, f"{base_name}_{snapshot_type}.csv")
                    snapshot_data.to_csv(snapshot_path, index=False)
        
        return path
    
    @property
    def columns(self) -> List[str]:
        """返回所有數據列（不包括 timestamp 和 symbol）"""
        return [c for c in self.data.columns if c not in ['timestamp', 'symbol']]
    
    @property
    def symbols(self) -> List[str]:
        """返回所有交易對"""
        return self.data['symbol'].unique().tolist()
    
    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """返回所有時間戳"""
        return pd.DatetimeIndex(self.data['timestamp'].unique())
    
    def info(self, verbose: bool = False) -> None:
        """打印 Panel 信息
        
        Args:
            verbose: 是否顯示詳細信息（包括快照數據）
        """
        from .console import print
        
        n_symbols = len(self.symbols)
        n_periods = len(self.timestamps)
        time_range = f"{self.timestamps.min().strftime('%Y-%m-%d')} to {self.timestamps.max().strftime('%Y-%m-%d')}"
        
        print(f"Panel: {len(self)} rows, {len(self.columns)} columns")
        print(f"  symbols={n_symbols}, periods={n_periods}, range={time_range}")
        
        if self.columns:
            nan_counts = {col: self.data[col].isna().sum() for col in self.columns}
            print(f"  NaN counts: {nan_counts}")
        
        # 顯示快照數據信息
        if self.snapshots:
            print(f"\n  Snapshot data available:")
            for snapshot_type in self.snapshots.keys():
                if snapshot_type == 'orderbook':
                    ob_symbols = list(self.snapshots['orderbook'].keys())
                    print(f"    - orderbook: {len(ob_symbols)} symbols ({', '.join(ob_symbols)})")
                elif isinstance(self.snapshots[snapshot_type], pd.DataFrame):
                    df = self.snapshots[snapshot_type]
                    print(f"    - {snapshot_type}: {len(df)} rows")
        
        # 詳細信息
        if verbose:
            print(f"\n  Column details:")
            for col in self.columns:
                dtype = self.data[col].dtype
                non_null = self.data[col].notna().sum()
                print(f"    {col:30s} {str(dtype):15s} non-null: {non_null}/{len(self)}")
            
            if 'orderbook' in self.snapshots:
                print(f"\n  Orderbook details:")
                for symbol, ob_df in self.snapshots['orderbook'].items():
                    n_bids = len(ob_df[ob_df['side'] == 'bid'])
                    n_asks = len(ob_df[ob_df['side'] == 'ask'])
                    print(f"    {symbol}: {n_bids} bids, {n_asks} asks")
    
    def describe(self) -> pd.DataFrame:
        """返回數據的統計摘要"""
        return self.data[self.columns].describe()
    
    def __repr__(self):
        n_symbols = len(self.symbols)
        n_periods = len(self.timestamps)
        time_range = f"{self.timestamps.min().strftime('%Y-%m-%d')} to {self.timestamps.max().strftime('%Y-%m-%d')}"
        
        snapshot_info = ""
        if self.snapshots:
            snapshot_types = ", ".join(self.snapshots.keys())
            snapshot_info = f", snapshots=[{snapshot_types}]"
        
        return (f"Panel({len(self)} rows, {len(self.columns)} cols, "
                f"{n_symbols} symbols, {n_periods} periods, {time_range}{snapshot_info})")
    
    def __len__(self):
        return len(self.data)
    
    def copy(self) -> 'Panel':
        """創建 Panel 的深拷貝"""
        new_panel = Panel(self.data.copy())
        new_panel.snapshots = {
            key: value.copy() if isinstance(value, pd.DataFrame) 
            else {k: v.copy() for k, v in value.items()} if isinstance(value, dict)
            else value
            for key, value in self.snapshots.items()
        }
        return new_panel
    
    def merge_snapshot(self, snapshot_type: str, snapshot_data: Any) -> None:
        """添加或更新快照數據
        
        Args:
            snapshot_type: 快照類型（如 'orderbook', 'tickers'）
            snapshot_data: 快照數據
        """
        self.snapshots[snapshot_type] = snapshot_data
    
    def drop_columns(self, columns: Union[str, List[str]]) -> 'Panel':
        """刪除指定列
        
        Args:
            columns: 要刪除的列名或列名列表
        """
        if isinstance(columns, str):
            columns = [columns]
        
        # 檢查列是否存在
        missing = [c for c in columns if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        
        # 不允許刪除 timestamp 和 symbol
        protected = [c for c in columns if c in ['timestamp', 'symbol']]
        if protected:
            raise ValueError(f"Cannot drop protected columns: {protected}")
        
        new_data = self.data.drop(columns=columns)
        new_panel = Panel(new_data)
        new_panel.snapshots = self.snapshots.copy()
        return new_panel
    
    def rename_columns(self, mapping: Dict[str, str]) -> 'Panel':
        """重命名列
        
        Args:
            mapping: 列名映射字典 {old_name: new_name}
        """
        # 不允許重命名 timestamp 和 symbol
        protected = [k for k in mapping.keys() if k in ['timestamp', 'symbol']]
        if protected:
            raise ValueError(f"Cannot rename protected columns: {protected}")
        
        new_data = self.data.rename(columns=mapping)
        new_panel = Panel(new_data)
        new_panel.snapshots = self.snapshots.copy()
        return new_panel