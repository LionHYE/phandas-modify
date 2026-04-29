"""
Sensitivity Analysis and Heatmap visualization for factor strategies.
Supports Parallel Processing (Multiprocessing) and Random Search.
Supports 2D (Seaborn) and 3D (Plotly) visualization.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
import os
import random
import math
from typing import Callable, Union, Tuple, List, Optional, Any, Dict, Iterable
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .backtest import Backtester

# 嘗試導入 Plotly，如果沒有安裝則會在調用 3D 繪圖時報錯
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# -----------------------------------------------------------------------------
# Worker Function
# -----------------------------------------------------------------------------
def _run_single_backtest(
    builder_func: Callable, 
    params: Dict[str, Any], 
    suppress_warnings: bool
) -> Dict[str, Any]:
    """
    Worker function to run a single backtest in a separate process.
    """
    result_row = params.copy()
    
    try:
        with warnings.catch_warnings():
            if suppress_warnings:
                warnings.simplefilter("ignore")
            
            bt = builder_func(**params)
            
            if not bt.metrics:
                bt.run().calculate_metrics()
            
            if bt.metrics:
                result_row.update(bt.metrics)
            else:
                result_row['error'] = "No metrics generated"

    except Exception as e:
        result_row['error'] = str(e)
    
    return result_row

# -----------------------------------------------------------------------------
# Sensitivity Result Class
# -----------------------------------------------------------------------------
class SensitivityResult:
    """
    Stores full results of a parameter sweep/random search and allows visualization.
    """

    def __init__(self, 
                 results_df: pd.DataFrame, 
                 param_names: List[str]):
        self.results = results_df
        self.param_names = param_names
        
    def _prepare_data(self, metric: str) -> pd.DataFrame:
        """Helper to clean data and calculate custom metrics if needed."""
        df = self.results.copy()
        
        if 'error' in df.columns:
            df = df[df['error'].isna()]

        if metric not in df.columns:
            try:
                df = df.fillna(0)
                df[metric] = df.eval(metric)
            except Exception as e:
                raise ValueError(f"Invalid metric or formula '{metric}'. Error: {e}")
        
        return df

    def get_best_params(self, metric: str = "sharpe_ratio") -> Dict:
        """
        Returns the parameters that yielded the best result for the given metric.
        """
        df = self._prepare_data(metric)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
        
        if df.empty:
            return {}

        best_idx = df[metric].idxmax()
        row = df.loc[best_idx]
        
        # Return all parameters plus the value
        result = {p: row[p] for p in self.param_names}
        result['value'] = row[metric]
        return result

    def plot(self, 
             metric: str = "sharpe_ratio",
             figsize: Tuple[int, int] = (10, 8), 
             cmap: str = "RdYlGn", 
             annot: bool = True, 
             fmt: str = ".2f",
             title: Optional[str] = None):
        """
        Automatically dispatches to 2D heatmap or 3D scatter based on parameter count.
        """
        n_params = len(self.param_names)
        
        if n_params == 2:
            self._plot_2d(metric, figsize, cmap, annot, fmt, title)
        elif n_params == 3:
            self._plot_3d(metric, title)
        else:
            print(f"Plotting not supported for {n_params} parameters directly. Access .results dataframe manually.")

    def _plot_2d(self, metric, figsize, cmap, annot, fmt, title):
        """Standard Seaborn Heatmap for 2 Parameters."""
        p1, p2 = self.param_names[0], self.param_names[1]
        
        try:
            df = self._prepare_data(metric)
            # Pivot table
            data = df.pivot(index=p2, columns=p1, values=metric)
        except ValueError as e:
            print(f"Cannot plot: {e}")
            return

        plt.figure(figsize=figsize)
        ax = sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, 
                         cbar_kws={'label': metric})
        ax.invert_yaxis()
        
        if title is None:
            title = f"Sensitivity Analysis (2D): {metric}"
            
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel(p1, fontsize=12)
        plt.ylabel(p2, fontsize=12)
        plt.tight_layout()
        plt.show()

    def _plot_3d(self, metric: str, title: Optional[str] = None):
        """Interactive Plotly 3D Scatter for 3 Parameters."""
        if not PLOTLY_AVAILABLE:
            print("Error: 'plotly' library is not installed. Please run `pip install plotly`.")
            return

        p1, p2, p3 = self.param_names[0], self.param_names[1], self.param_names[2]
        
        try:
            df = self._prepare_data(metric)
        except ValueError as e:
            print(f"Cannot plot: {e}")
            return
            
        if title is None:
            title = f"Sensitivity Analysis (3D): {metric} by {p1}, {p2}, {p3}"

        # 創建 3D 散點圖
        # x, y, z 是三個參數
        # color 是回測績效 (如 Sharpe)
        fig = px.scatter_3d(
            df, 
            x=p1, y=p2, z=p3,
            color=metric,
            color_continuous_scale="RdYlGn",  # 類似你原本的 cmap
            title=title,
            hover_data=self.param_names + [metric],
            labels={
                p1: f"{p1} (X)",
                p2: f"{p2} (Y)",
                p3: f"{p3} (Z)",
                metric: metric
            }
        )
        
        # 優化標記大小，讓它看起來更像是一個體積圖
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        
        # 設定佈局
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=50),
            scene=dict(
                xaxis_title=p1,
                yaxis_title=p2,
                zaxis_title=p3
            )
        )
        
        fig.show()

# -----------------------------------------------------------------------------
# Main Execution Function
# -----------------------------------------------------------------------------
def run_parameter_sweep(
    strategy_builder: Callable[..., Backtester],
    param_grid: Dict[str, Iterable[Any]],
    n_iter: Optional[int] = None,
    show_progress: bool = True,
    suppress_warnings: bool = True,
    n_jobs: int = -1
) -> SensitivityResult:
    """
    Executes a parameter sweep. 
    Supports 2 or more parameters.
    """
    
    param_names = list(param_grid.keys())
    if len(param_names) < 2:
        raise ValueError("Requires at least 2 parameters for sensitivity analysis.")
    
    # 確保所有輸入轉為 list
    values_list = [list(param_grid[k]) for k in param_names]
    
    # 計算全量組合總數
    total_possible_combinations = math.prod([len(v) for v in values_list])
    
    tasks_params = []
    
    # --- 決策：Grid Search vs Random Search ---
    if n_iter is None or n_iter >= total_possible_combinations:
        # 模式 A: Grid Search (全量)
        mode_name = "Grid Search"
        combinations = list(itertools.product(*values_list))
        tasks_params = [dict(zip(param_names, combo)) for combo in combinations]
    else:
        # 模式 B: Random Search (隨機取樣)
        mode_name = f"Random Search (n={n_iter})"
        seen_combinations = set()
        
        with tqdm(total=n_iter, desc="Generating Samples", disable=not show_progress) as pbar:
            while len(tasks_params) < n_iter:
                combo_tuple = tuple(random.choice(v) for v in values_list)
                
                if combo_tuple not in seen_combinations:
                    seen_combinations.add(combo_tuple)
                    tasks_params.append(dict(zip(param_names, combo_tuple)))
                    pbar.update(1)
                
                if len(seen_combinations) >= total_possible_combinations:
                    break

    total_tasks = len(tasks_params)
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    print(f"Starting {mode_name}: {total_tasks} tasks (Params: {param_names}) using {n_jobs} cores.")
    
    results = []

    # --- 執行部分 ---
    if n_jobs == 1:
        iterator = tqdm(tasks_params, desc="Backtesting", unit="run", dynamic_ncols=True) if show_progress else tasks_params
        for params in iterator:
            res = _run_single_backtest(strategy_builder, params, suppress_warnings)
            if 'error' in res and show_progress:
                tqdm.write(f"[Error] {params}: {res['error']}")
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_run_single_backtest, strategy_builder, p, suppress_warnings): p 
                for p in tasks_params
            }
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=total_tasks, desc="Parallel Backtesting", unit="run", dynamic_ncols=True)
            
            for future in iterator:
                try:
                    res = future.result()
                    if 'error' in res and show_progress:
                        tqdm.write(f"[Error]: {res.get('error')}")
                    results.append(res)
                except Exception as exc:
                    tqdm.write(f"Process generated an exception: {exc}")

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("No results generated.")
        
    return SensitivityResult(df, param_names)