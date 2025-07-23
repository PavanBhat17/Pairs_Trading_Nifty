# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

def load_and_inspect_data(parquet_file_path):
    """
    Load a parquet file containing IV data, print a summary, restrict to trading hours, and perform exploratory data analysis.
    Args:
        parquet_file_path (str): Path to the parquet file.
    Returns:
        pd.DataFrame: DataFrame indexed by datetime, restricted to trading hours.
    """
    import pandas as pd
    from IPython.display import display
    df = pd.read_parquet(parquet_file_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        datetime_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_col = col
                break
        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    print("\nData Overview:")
    print(f"Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {df.index.name if df.index.name else 'No name'}")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    try:
        display(df.describe().T)
    except Exception:
        print(df.describe().T)
    print("\nFirst 3 rows:")
    try:
        display(df.head(3))
    except Exception:
        print(df.head(3))
    print("\nLast 3 rows:")
    try:
        display(df.tail(3))
    except Exception:
        print(df.tail(3))
    try:
        df = df.between_time('09:15:00', '15:30:00')
        print(f"\nAfter restricting to trading hours: {df.shape[0]} rows remain.")
    except Exception as e:
        print(f"Could not restrict to trading hours: {e}")
    return df

def handle_missing_data_ffill_garch(df, max_ffill_days: int = 3):
    """
    Impute missing values in the IV data using forward fill for short gaps and GARCH(1,1) for longer gaps. Falls back to linear interpolation if GARCH fails.
    Args:
        df (pd.DataFrame): DataFrame with missing values.
        max_ffill_days (int, optional): Maximum days for forward fill. Defaults to 3.
    Returns:
        pd.DataFrame: DataFrame with missing values imputed and an 'is_interpolated' column added.
    """
    import pandas as pd
    import numpy as np
    from arch import arch_model
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    tz = df.index.tz if hasattr(df.index, 'tz') else None
    full_idx = pd.date_range(df.index.min().floor('T'),
                             df.index.max().ceil('T'), 
                             freq='T', tz=tz)
    full_idx = full_idx[full_idx.dayofweek < 5]
    df_full = df.reindex(full_idx)
    raw_nan = df_full.isna().any(axis=1)
    mins_per_session = 375
    ffill_limit = max_ffill_days * mins_per_session
    df_tmp = df_full.fillna(method='ffill', limit=ffill_limit)
    df_filled = df_tmp.copy()
    for col in ['nifty', 'banknifty']:
        if col not in df_tmp.columns:
            print(f"Column '{col}' not found in DataFrame. Skipping GARCH imputation for this column.")
            continue
        series = df_tmp[col]
        if series.isna().any():
            print(f"Applying GARCH imputation to {col.upper()}...")
            available_data = series.dropna()
            if len(available_data) < 100:
                df_filled[col] = series.interpolate(method='linear')
                continue
            returns = available_data.pct_change().dropna() * 100
            try:
                garch_model = arch_model(returns, vol='GARCH', p=1, q=1, rescale=False)
                garch_fitted = garch_model.fit(disp='off')
                last_level = available_data.iloc[-1]
                last_return = returns.iloc[-1]
                missing_mask = series.isna()
                gap_labels = (missing_mask != missing_mask.shift()).cumsum()[missing_mask]
                for _, gap_idx in series[missing_mask].groupby(gap_labels).groups.items():
                    if len(gap_idx) <= ffill_limit:
                        continue
                    gap_length = len(gap_idx)
                    forecast = garch_fitted.forecast(horizon=min(gap_length, 1000), reindex=False)
                    vol_forecast = np.sqrt(forecast.variance.values[-1])
                    np.random.seed(42)
                    omega = garch_fitted.params['omega']
                    alpha = garch_fitted.params['alpha[1]']
                    beta = garch_fitted.params['beta[1]']
                    current_level = last_level
                    current_vol = vol_forecast[0] if len(vol_forecast) > 0 else np.std(returns)
                    imputed_levels = []
                    for i in range(gap_length):
                        if i > 0:
                            current_vol = np.sqrt(omega + alpha * (prev_return**2) + beta * (current_vol**2))
                        random_return = np.random.normal(0, current_vol)
                        mean_reversion = 0.999
                        long_term_mean = available_data.mean()
                        current_level = current_level * mean_reversion + (1-mean_reversion) * long_term_mean
                        current_level = current_level * (1 + random_return/100)
                        current_level = max(current_level, 0.01)
                        imputed_levels.append(current_level)
                        prev_return = random_return
                    df_filled.loc[gap_idx, col] = imputed_levels
            except Exception as e:
                print(f"GARCH fitting failed for {col}, using linear interpolation: {e}")
                df_filled[col] = series.interpolate(method='linear')
    try:
        df_out = df_filled.between_time('09:15', '15:30')
    except Exception as e:
        print(f"Could not restrict to trading hours in imputation: {e}")
        df_out = df_filled
    df_out['is_interpolated'] = raw_nan.reindex(df_out.index, fill_value=False).astype(np.int8)
    return df_out

def calculate_cointegration_signals(df, lookback_window=252):
    """
    Rolling cointegration-based signal generation for pairs trading. For each window, test for cointegration and estimate hedge ratio and residuals. Signals are generated only when cointegration is detected.
    Args:
        df (pd.DataFrame): DataFrame with 'nifty' and 'banknifty' columns.
        lookback_window (int): Rolling window size for cointegration test and hedge ratio estimation.
    Returns:
        pd.DataFrame: DataFrame with cointegration signals, hedge ratio, and residuals.
    """
    df['spread'] = df['banknifty'] - df['nifty']
    df['coint_signal'] = 0.0
    df['hedge_ratio'] = np.nan
    df['residual'] = np.nan
    df['cointegrated'] = False
    total_iterations = len(df) - lookback_window
    print(f"Calculating cointegration signals for {total_iterations:,} observations...")
    for i in tqdm(range(lookback_window, len(df)), desc="Cointegration Analysis", unit="obs", ncols=100):
        lookback_data = df.iloc[i-lookback_window:i]
        try:
            score, pvalue, _ = coint(lookback_data['banknifty'], lookback_data['nifty'])
            if pvalue < 0.05:
                df.iloc[i, df.columns.get_loc('cointegrated')] = True
                y = lookback_data['banknifty']
                x = lookback_data['nifty']
                model = OLS(y, x).fit()
                hedge_ratio = model.params[0]
                current_residual = df.iloc[i]['banknifty'] - hedge_ratio * df.iloc[i]['nifty']
                residual_series = lookback_data['banknifty'] - hedge_ratio * lookback_data['nifty']
                residual_mean = residual_series.mean()
                residual_std = residual_series.std()
                if residual_std > 0:
                    signal = (current_residual - residual_mean) / residual_std
                    df.iloc[i, df.columns.get_loc('coint_signal')] = signal
                    df.iloc[i, df.columns.get_loc('hedge_ratio')] = hedge_ratio
                    df.iloc[i, df.columns.get_loc('residual')] = current_residual
        except Exception:
            continue
    cointegrated_periods = df['cointegrated'].sum()
    print(f"Cointegration analysis complete. Found {cointegrated_periods:,} cointegrated periods.")
    df['coint_signal'] = df['coint_signal'].fillna(0.0)
    df['hedge_ratio'] = df['hedge_ratio'].fillna(method='ffill')
    df['cointegrated'] = df['cointegrated'].fillna(False)
    return df

def generate_trading_signals(df, entry_threshold=1.5, exit_threshold=0.3):
    """
    Generate trading signals and positions based on cointegration signals. Only generate signals when pairs are cointegrated.
    Args:
        df (pd.DataFrame): DataFrame with 'coint_signal' and 'cointegrated' columns.
        entry_threshold (float): Threshold for entering trades.
        exit_threshold (float): Threshold for exiting trades.
    Returns:
        pd.DataFrame: DataFrame with position, signal, and position_change columns added.
    """
    signal = df['coint_signal']
    cointegrated = df['cointegrated']
    long_entry = (signal < -entry_threshold) & cointegrated
    short_entry = (signal > entry_threshold) & cointegrated
    exit_signal = (signal.abs() < exit_threshold) | (~cointegrated)
    raw_dir = np.where(exit_signal, 0,
                      np.where(long_entry, 1,
                              np.where(short_entry, -1, np.nan)))
    df['position'] = (pd.Series(raw_dir, index=df.index)
                     .ffill()
                     .fillna(0)
                     .astype(np.int8))
    eod = df.index.time == pd.to_datetime('15:30:00').time()
    df.loc[eod, 'position'] = 0
    df['signal'] = df['position'].diff().fillna(df['position']).astype(np.int8)
    df['position_change'] = df['signal']
    return df

def calculate_pnl(df, trade_cost_bps=0.0325):
    """
    Calculate trade-by-trade and cumulative profit and loss (P&L) for the strategy. P&L is computed for each trade based on spread change, average TTE, trade direction, and transaction costs.
    Args:
        df (pd.DataFrame): DataFrame with 'position', 'spread', and 'tte' columns.
        trade_cost_bps (float, optional): Transaction cost per trade. Defaults to 0.0325.
    Returns:
        pd.DataFrame: DataFrame with 'trade_pnl' and 'cumulative_pnl' columns added.
    """
    df['trade_pnl'] = 0.0
    cumulative_pnl = 0.0
    open_pos = 0
    entry_spread = entry_tte = entry_dir = None
    for idx, row in df.iterrows():
        curr_pos = row['position']
        enter = (open_pos == 0 and curr_pos != 0) or \
                (open_pos != 0 and np.sign(curr_pos) != np.sign(open_pos))
        exit_ = (open_pos != 0 and curr_pos == 0) or \
                (open_pos != 0 and np.sign(curr_pos) != np.sign(open_pos))
        if exit_:
            spread_change = row['spread'] - entry_spread
            avg_tte = ((entry_tte + row['tte']) / 2.0) ** 0.7
            pnl = spread_change * avg_tte * entry_dir
            pnl -= trade_cost_bps
            df.at[idx, 'trade_pnl'] = pnl
            cumulative_pnl += pnl
            open_pos = 0
        if enter:
            entry_spread = row['spread']
            entry_tte = row['tte']
            entry_dir = curr_pos
            open_pos = curr_pos
    df['cumulative_pnl'] = df['trade_pnl'].cumsum()
    return df

def calculate_performance_metrics(df):
    """
    Compute key performance metrics for the strategy, including total P&L, number of trades, Sharpe ratio, drawdown, and win rate.
    Args:
        df (pd.DataFrame): DataFrame with 'cumulative_pnl' and 'trade_pnl' columns.
    Returns:
        dict: Dictionary of performance metrics.
    """
    total_pnl = df['cumulative_pnl'].iloc[-1]
    num_trades = (df['trade_pnl'] != 0).sum()
    daily_pnl = df['cumulative_pnl'].resample('D').last().diff().dropna()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() != 0 else 0)
    equity = df['cumulative_pnl']
    running_max = equity.cummax()
    drawdown = equity - running_max
    drawdown_pct = drawdown / running_max.replace(0, np.nan)
    max_dd = drawdown.min()
    max_dd_pct = drawdown_pct.min()
    trade_pnls = df.loc[df['trade_pnl'] != 0, 'trade_pnl']
    win_rate = (trade_pnls > 0).mean() if len(trade_pnls) else 0
    return {
        'Total P/L': total_pnl,
        'Number of Trades': num_trades,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (abs)': max_dd,
        'Max Drawdown (%)': max_dd_pct,
        'Win Rate': win_rate
    }

def plot_results(df):
    """
    Plot the main results of the cointegration strategy, including spread, cointegration signal, positions, cumulative P&L, and drawdown. Saves the plot as a PNG in the output directory.
    Args:
        df (pd.DataFrame): DataFrame with strategy results and metrics columns.
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    output_dir = 'Cointegration-Strategy-Plots'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'cointegration_strategy_results_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    # Plot 1: Spread
    axes[0].plot(df.index, df['spread'], label='Spread', alpha=0.7)
    axes[0].set_ylabel('Spread')
    axes[0].set_title('Bank Nifty - Nifty IV Spread')
    axes[0].legend(loc='upper left')
    # Plot 2: Cointegration Signal
    axes[1].plot(df.index, df['coint_signal'], label='Cointegration Signal', color='purple', alpha=0.7)
    axes[1].axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Entry Threshold')
    axes[1].axhline(y=-1.5, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Exit Threshold')
    axes[1].axhline(y=-0.3, color='green', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Cointegration Signal')
    axes[1].set_title('Standardized Residual (Cointegration Z-Score)')
    axes[1].legend()
    # Plot 3: Cointegrated Periods
    axes[2].plot(df.index, df['cointegrated'].astype(int), label='Cointegrated', color='orange', alpha=0.7)
    axes[2].set_ylabel('Cointegrated')
    axes[2].set_title('Cointegration Regime (1=Cointegrated)')
    axes[2].legend()
    # Plot 4: Position
    axes[3].step(df.index, df['position'], label='Position', color='blue', where='post')
    axes[3].set_ylabel('Position')
    axes[3].set_title('Trading Position Over Time')
    axes[3].legend()
    # Plot 5: Cumulative PnL
    axes[4].plot(df.index, df['cumulative_pnl'], label='Cumulative P/L', color='green', linewidth=1.5)
    axes[4].set_ylabel('Cumulative P/L')
    axes[4].set_title('Strategy Performance (Cumulative P&L)')
    axes[4].legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filepath}")
    plt.show()

def main():
    """
    Run the full cointegration strategy pipeline: load data, preprocess, calculate cointegration signals, generate trading signals, evaluate, and plot results. Prints performance metrics and saves plots.
    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame with results and dictionary of performance metrics.
    """
    try:
        df = load_and_inspect_data('data.parquet')
        df = handle_missing_data_ffill_garch(df)
        print(f"After handling missing data: {df.shape[0]} rows")
        df = calculate_cointegration_signals(df, lookback_window=252)
        df = generate_trading_signals(df, entry_threshold=1.5, exit_threshold=0.3)
        df = calculate_pnl(df)
        mts = calculate_performance_metrics(df)
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        for k, v in mts.items():
            if isinstance(v, float):
                print(f'{k:22}: {v:,.6f}')
            else:
                print(f'{k:22}: {v}')
        plot_results(df)
        return df, mts
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    df_results, performance = main() 