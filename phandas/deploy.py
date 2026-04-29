"""Deployment module for generating live trading signals."""

import json
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
from .core import Factor
from .panel import Panel  # 引用新的 Panel class

def deploy_factor(
    panel: Union[Panel, Dict],
    strategy_factor: Factor,
    neutralization: str = "market",
    lookback_window: int = 1,
    debug: bool = True
) -> str:
    """
    將策略因子轉換為即時交易權重 JSON（適配新版 Panel 與 Unclosed Candle）。
    
    Parameters
    ----------
    panel : Panel
        新版市場數據 Panel 物件。
    strategy_factor : Factor
        計算好的策略因子。
    neutralization : str, default "market"
        "market": 多空對沖 (權重總和為 0，絕對值總和為 1)。
        "none": 原始訊號權重 (僅做總資金歸一化)。
    lookback_window : int, default 1
        如果最新時間點 (Latest) 的數據為 NaN，允許向前回溯多少根 K 線尋找有效訊號。
        設為 0 代表嚴格只看當下。

    Returns
    -------
    str
        包含 "幣種: 權重(%)" 的 JSON 字串。
    """
    
    # 1. 安全檢查
    if strategy_factor.data.empty:
        if debug: print("[Deploy] Error: Strategy factor data is empty.")
        return json.dumps({})

    # 2. 獲取數據與時間點
    df = strategy_factor.data.sort_values('timestamp')
    latest_ts = df['timestamp'].max()
    
    if debug:
        print(f"[Deploy] Latest signal timestamp: {latest_ts}")

    # 3. 提取最新截面 (Snapshot)
    # 這裡加入 ffill 邏輯，防止因為未收盤 K 線導致的單點 NaN
    # 我們取最後 lookback_window + 1 行，然後 fillna，再取最後一行
    
    # 先過濾出最後幾個時間點
    available_timestamps = df['timestamp'].unique()
    if len(available_timestamps) > lookback_window + 1:
        start_ts = available_timestamps[-(lookback_window + 1)]
        recent_df = df[df['timestamp'] >= start_ts].copy()
    else:
        recent_df = df.copy()

    # 對每個 symbol 進行 ffill (沿用舊訊號填補最新空的訊號)
    recent_df['factor'] = recent_df.groupby('symbol')['factor'].ffill()
    
    # 再次獲取最新時間點的數據 (此時 NaN 應該已被填補)
    subset = recent_df[recent_df['timestamp'] == latest_ts].set_index('symbol')['factor']
    
    # 移除仍為 NaN 的幣種 (代表連前幾根都沒數據)
    subset = subset.dropna()

    if subset.empty:
        if debug: print(f"[Deploy] Warning: No valid signals found at {latest_ts} (after fill).")
        return json.dumps({})

    # 4. 計算權重邏輯
    weights = {}
    
    if neutralization.lower() == "market":
        # 實作去均值與歸一化 (手動實作以確保透明度)
        demeaned = subset - subset.mean()
        abs_sum = demeaned.abs().sum()
        
        if abs_sum > 1e-10:
            weights = (demeaned / abs_sum).to_dict()
        else:
            if debug: print("[Deploy] Warning: Signal strength (abs_sum) is too low for market neutral.")
            weights = {}
            
    elif neutralization.lower() == "top_k":
        # 範例：只做多最強的 3 隻
        top_k = 3
        top_coins = subset.nlargest(top_k)
        if not top_coins.empty:
            weight_per_coin = 1.0 / len(top_coins)
            weights = {sym: weight_per_coin for sym in top_coins.index}
            
    else:
        # "none" 模式：保留原始方向，僅做絕對值歸一化
        abs_sum = subset.abs().sum()
        if abs_sum > 1e-10:
            weights = (subset / abs_sum).to_dict()
        else:
            weights = {}

    # 5. 格式化輸出
    # 轉換為百分比，過濾微小份額
    output_dict = {
        symbol: round(weight * 100, 4)
        for symbol, weight in weights.items()
        if abs(weight) > 1e-4  # 過濾小於 0.0001% 的倉位
    }
    
    if debug:
        print(f"[Deploy] Generated weights for {len(output_dict)} symbols.")

    return json.dumps(output_dict, indent=4, sort_keys=True)