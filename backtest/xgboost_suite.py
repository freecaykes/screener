import joblib
import pandas as pd

from .backtest import Backtester, print_backtest_results


def xgboost_run_simulation():
    # Load your trained model
    model = joblib.load("xgboost_stock_delta_model.pkl")

    # Load historical data (with indicators pre-computed)
    data = pd.read_parquet("data/aapl_with_indicators.parquet")

    backtester = Backtester(initial_capital=100_000, commission=0.001)

    result = backtester.run(
        data=data,
        strategy_func=xgboost_news_strategy,
        model=model,
        sentiment_series=data.get("sentiment_score", pd.Series(0.0, index=data.index))
    )

    print_backtest_results(result)


def xgboost_news_strategy(df: pd.DataFrame, model, **kwargs) -> pd.DataFrame:
    """
    df must have OHLCV + any needed columns.
    This function should simulate what your agent does during live trading.
    """
    signals = pd.DataFrame(index=df.index, columns=['signal'], data=0)

    for i in range(20, len(df)):  # need enough history for indicators
        try:
            # Replicate compute_indicators logic here (simplified)
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # Example: Use your trained model
            features = {
                "sentiment_score": kwargs.get("sentiment_series", pd.Series(0.0, index=df.index)).iloc[i],
                "RSI_14": row.get("RSI_14", 50),
                "price_to_ema21": row.get("price_to_ema21", 1.0),
                "pullback_buy_setup": row.get("pullback_buy_setup", 0),
                "vix_current": row.get("vix_current", 20),
            }

            X = pd.DataFrame([features])
            pred_delta = model.predict(X)[0]

            # Simple rule: Strong positive predicted move + pullback setup = BUY
            if pred_delta > 0.8 and row.get("pullback_buy_setup", 0) == 1:
                signals.iloc[i] = 1  # Buy
            elif pred_delta < -0.8:
                signals.iloc[i] = -1  # Sell

        except:
            continue

    return signals
