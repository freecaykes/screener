# backtest/backtester.py
from dataclasses import dataclass
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    equity_curve: pd.Series
    trades: pd.DataFrame
    metadata: dict


class Backtester:
    def __init__(self,
                 initial_capital: float = 100_000,
                 commission: float = 0.001,  # 0.1%
                 slippage: float = 0.001,
                 max_position_size: float = 0.25):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size

    def run_backtest(self,
                     data: pd.DataFrame,
                     strategy_func: Callable,
                     **strategy_kwargs) -> BacktestResult:
        """Standard historical backtest"""
        return self._run_simulation(data, strategy_func, **strategy_kwargs)

    def run_walk_forward(self,
                         data: pd.DataFrame,
                         strategy_func: Callable,
                         train_period: int = 252 * 2,  # 2 years training
                         test_period: int = 63,  # ~3 months forward testing
                         step_size: int = 63,  # How much to roll forward
                         retrain: bool = True,  # Retrain model each window?
                         **strategy_kwargs) -> Dict[str, Any]:
        """
        Walk-Forward Testing (most realistic)
        - Trains on past data
        - Tests on future unseen data
        - Rolls forward
        """
        results = []
        equity_curves = []
        all_trades = []
        start_idx = train_period

        print(f"Starting Walk-Forward Analysis | Train: {train_period} days | Test: {test_period} days")

        while start_idx + test_period <= len(data):
            train_end = start_idx
            test_end = start_idx + test_period

            train_data = data.iloc[:train_end].copy()
            test_data = data.iloc[train_end:test_end].copy()

            window_start = data.index[train_end]
            window_end = data.index[test_end - 1]

            print(f"Window {len(results) + 1}: Training until {window_start.date()} | Testing {window_end.date()}")

            # Retrain model if needed (important for your XGBoost + sentiment strategy)
            if retrain and 'model' in strategy_kwargs:
                print("   → Retraining model...")
                # You can pass a retrain function or handle inside strategy_func

            # Run test period
            result = self._run_simulation(test_data, strategy_func, **strategy_kwargs)
            results.append(result)
            equity_curves.append(result.equity_curve)
            all_trades.extend(result.trades.to_dict('records'))

            start_idx += step_size

        # Combine results
        combined_equity = pd.concat(equity_curves)
        combined_trades = pd.DataFrame(all_trades)

        # Calculate overall metrics
        total_return = (combined_equity.iloc[-1] / combined_equity.iloc[0]) - 1
        returns = combined_equity.pct_change().dropna()
        cagr = (combined_equity.iloc[-1] / combined_equity.iloc[0]) ** (252 / len(returns)) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        downside = returns[returns < 0]
        sortino = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        max_dd = (combined_equity / combined_equity.cummax() - 1).min()

        win_rate = (combined_trades['pnl_pct'] > 0).mean() if not combined_trades.empty else 0

        return {
            "walk_forward_results": results,
            "overall": BacktestResult(
                total_return=total_return,
                cagr=cagr,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=0,  # Can be calculated if needed
                num_trades=len(combined_trades),
                avg_trade_return=combined_trades['pnl_pct'].mean() if not combined_trades.empty else 0,
                equity_curve=combined_equity,
                trades=combined_trades,
                metadata={"type": "walk_forward", "windows": len(results)}
            )
        }

    def _run_simulation(self,
                        df: pd.DataFrame,
                        strategy_func: Callable,
                        **kwargs) -> BacktestResult:
        """Core simulation engine used by both backtest and walk-forward"""
        df = df.copy()
        signals = strategy_func(df, **kwargs)
        df = df.join(signals)

        equity = [self.initial_capital]
        trades = []
        position = 0.0
        entry_price = 0.0
        entry_date = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            signal = row.get('signal', 0)

            # Exit
            if position > 0 and signal == -1:
                exit_val = position * row['Close'] * (1 - self.slippage - self.commission)
                pnl = exit_val - (position * entry_price)
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row.name,
                    'entry_price': entry_price,
                    'exit_price': row['Close'],
                    'pnl_pct': (row['Close'] / entry_price - 1) * 100,
                    'pnl_dollar': pnl
                })
                equity.append(equity[-1] + pnl)
                position = 0.0

            # Entry
            elif position == 0 and signal == 1:
                capital = equity[-1]
                position_value = capital * self.max_position_size
                entry_price = row['Close'] * (1 + self.slippage)
                position = position_value / entry_price
                entry_date = row.name
                equity.append(equity[-1] - position_value * self.commission)

            # Hold
            else:
                if position > 0:
                    equity.append(equity[-1] * (row['Close'] / prev['Close']))
                else:
                    equity.append(equity[-1])

        equity_curve = pd.Series(equity, index=df.index[:len(equity)])

        # Calculate metrics (same as before)
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0

        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        downside = returns[returns < 0]
        sortino = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        max_dd = (equity_curve / equity_curve.cummax() - 1).min()

        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl_pct'] > 0).mean() if not trades_df.empty else 0

        return BacktestResult(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=0,  # implement if needed
            num_trades=len(trades),
            avg_trade_return=trades_df['pnl_pct'].mean() if not trades_df.empty else 0,
            equity_curve=equity_curve,
            trades=trades_df,
            metadata={"type": "single_period"}
        )


def print_backtest_results(result: BacktestResult):
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return     : {result.total_return * 100:8.2f}%")
    print(f"CAGR             : {result.cagr * 100:8.2f}%")
    print(f"Sharpe Ratio     : {result.sharpe_ratio:8.2f}")
    print(f"Sortino Ratio    : {result.sortino_ratio:8.2f}")
    print(f"Max Drawdown     : {result.max_drawdown * 100:8.2f}%")
    print(f"Win Rate         : {result.win_rate * 100:8.2f}%")
    print(f"Profit Factor    : {result.profit_factor:8.2f}")
    print(f"Number of Trades : {result.num_trades:8d}")
    print("=" * 60)
