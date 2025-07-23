# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from arch import arch_model

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
    # Ensure index is DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to find a datetime column
        datetime_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_col = col
                break
        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, inplace=True)
        else:
            # Fallback: try to convert index
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
    # Restrict to trading hours
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

    # Ensure index is DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Try to preserve timezone if present
    tz = df.index.tz if hasattr(df.index, 'tz') else None
    # Create a full minute index for all trading days
    full_idx = pd.date_range(df.index.min().floor('T'),
                             df.index.max().ceil('T'), 
                             freq='T', tz=tz)
    # Only keep weekdays
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
    # Mark imputed rows
    try:
        df_out = df_filled.between_time('09:15', '15:30')
    except Exception as e:
        print(f"Could not restrict to trading hours in imputation: {e}")
        df_out = df_filled
    df_out['is_interpolated'] = raw_nan.reindex(df_out.index, fill_value=False).astype(np.int8)
    return df_out


def calculate_spread_and_zscore(df, lookback_window=200):
    """
    Compute the spread between Bank Nifty IV and Nifty IV, and calculate the rolling z-score of the spread.

    Args:
        df (pd.DataFrame): DataFrame with 'banknifty' and 'nifty' columns.
        lookback_window (int, optional): Window size for rolling mean and std. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame with spread, rolling mean, std, and z-score columns added.
    """
    # Calculate the spread between Bank Nifty IV and Nifty IV
    df['spread'] = df['banknifty'] - df['nifty']

    # Calculate rolling mean and std of the spread
    df['spread_mean'] = df['spread'].rolling(
        window=lookback_window, min_periods=lookback_window).mean()
    df['spread_std'] = df['spread'].rolling(
        window=lookback_window, min_periods=lookback_window).std()

    # Calculate the z-score (standardized spread)
    z_raw = (df['spread'] - df['spread_mean']) / df['spread_std'].replace(0, np.nan)
    df['z_score'] = z_raw.fillna(0.0)  # Fill initial NaNs with 0 for stability

    # Optional: Print a quick summary for debugging
    print(f"\nZ-Score Calculation: {df['z_score'].min():.2f} to {df['z_score'].max():.2f}")
    print(f"Spread mean (last): {df['spread_mean'].iloc[-1]:.4f}, std (last): {df['spread_std'].iloc[-1]:.4f}")

    return df


def generate_trading_signals(df, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate trading signals and positions based on z-score thresholds. Entry and exit signals are determined by comparing the z-score to specified thresholds.

    Args:
        df (pd.DataFrame): DataFrame with 'z_score' column.
        entry_threshold (float, optional): Z-score threshold for entering trades. Defaults to 2.0.
        exit_threshold (float, optional): Z-score threshold for exiting trades. Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame with position, signal, and position_change columns added.
    """
    z = df['z_score']

    # Entry and exit conditions
    long_entry   = (z < -entry_threshold)
    short_entry  = (z >  entry_threshold)
    exit_signal  = (z.abs() < exit_threshold)

    # Raw direction: 1 for long, -1 for short, 0 for exit, np.nan otherwise
    raw_dir = np.where(exit_signal, 0,
              np.where(long_entry,   1,
              np.where(short_entry, -1, np.nan)))

    # Forward-fill positions, fill initial NaNs with 0 (flat)
    df['position'] = (pd.Series(raw_dir, index=df.index)
                        .ffill()
                        .fillna(0)
                        .astype(np.int8))

    # Flatten position at end of day (EOD)
    eod = df.index.time == pd.to_datetime('15:30:00').time()
    df.loc[eod, 'position'] = 0

    # Signal: change in position (entry/exit)
    df['signal'] = df['position'].diff().fillna(df['position']).astype(np.int8)
    df['position_change'] = df['signal']  # For metrics

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

        # Entry: open new position or flip direction
        enter = (open_pos == 0 and curr_pos != 0) or \
                (open_pos != 0 and np.sign(curr_pos) != np.sign(open_pos))

        # Exit: close position or flip direction
        exit_ = (open_pos != 0 and curr_pos == 0) or \
                (open_pos != 0 and np.sign(curr_pos) != np.sign(open_pos))

        if exit_:
            # Calculate P/L for the closed trade
            spread_change = row['spread'] - entry_spread
            avg_tte = ((entry_tte + row['tte']) / 2.0) ** 0.7
            pnl = spread_change * avg_tte * entry_dir
            pnl -= trade_cost_bps
            df.at[idx, 'trade_pnl'] = pnl
            cumulative_pnl += pnl
            open_pos = 0

        if enter:
            # Record entry details
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

    # Daily P/L for Sharpe calculation
    daily_pnl = df['cumulative_pnl'].resample('D').last().diff().dropna()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() != 0 else 0)

    # Drawdown
    equity = df['cumulative_pnl']
    running_max = equity.cummax()
    drawdown = equity - running_max
    drawdown_pct = drawdown / running_max.replace(0, np.nan)
    max_dd = drawdown.min()
    max_dd_pct = drawdown_pct.min()

    # Win rate
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
    Plot the main results of the strategy, including spread, z-score, positions, cumulative P&L, and drawdown. Saves the plot as a PNG in the output directory.

    Args:
        df (pd.DataFrame): DataFrame with strategy results and metrics columns.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    output_dir = 'Z-Score-Strategy-Plots'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'zscore_strategy_results_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)

    fig, axes = plt.subplots(4, 1, figsize=(15, 16))

    # Plot 1: Spread and Z-Score with trade markers
    axes[0].plot(df.index, df['spread'], label='Spread', alpha=0.7)
    axes[0].set_ylabel('Spread')
    axes[0].set_title('Bank Nifty - Nifty IV Spread')
    axes[0].legend(loc='upper left')

    ax0_twin = axes[0].twinx()
    ax0_twin.plot(df.index, df['z_score'], color='red', label='Z-Score', alpha=0.7)
    ax0_twin.axhline(y=2, color='r', linestyle='--', alpha=0.5)
    ax0_twin.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
    ax0_twin.set_ylabel('Z-Score')
    ax0_twin.legend(loc='upper right')

    # Mark trade entries/exits
    entries = df[df['signal'] != 0]
    axes[0].scatter(entries.index, entries['spread'], marker='o', color='black', s=30, label='Trade Entry/Exit', zorder=5)
    axes[0].legend(loc='upper left')

    # Plot 2: Positions
    axes[1].step(df.index, df['position'], label='Position', color='orange', where='post')
    axes[1].set_ylabel('Position')
    axes[1].set_title('Trading Positions')
    axes[1].legend()

    # Plot 3: Cumulative P/L
    axes[2].plot(df.index, df['cumulative_pnl'], label='Cumulative P/L', color='green')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Cumulative P/L')
    axes[2].set_title('Strategy Performance')
    axes[2].legend()

    # Plot 4: Drawdown
    running_max = df['cumulative_pnl'].cummax()
    drawdown = df['cumulative_pnl'] - running_max
    axes[3].fill_between(df.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    axes[3].set_ylabel('Drawdown')
    axes[3].set_title('Drawdown Over Time')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
    plt.show()


def main():
    """
    Run the full z-score strategy pipeline: load data, preprocess, calculate signals, evaluate, and plot results. Prints performance metrics and saves plots.

    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame with results and dictionary of performance metrics.
    """
    # Step 1: Load and inspect data
    df = load_and_inspect_data('data.parquet')

    # Step 2: Handle missing data
    df = handle_missing_data_ffill_garch(df)

    # Step 3: Calculate spread and z-score
    df = calculate_spread_and_zscore(df, lookback_window=100)

    # Step 4: Generate trading signals
    df = generate_trading_signals(df, entry_threshold=2.0, exit_threshold=0.5)
    
    # Step 5: Calculate P&L
    df = calculate_pnl(df)

    # Step 6: Calculate performance metrics
    mts = calculate_performance_metrics(df)

    # Print performance metrics
    print("\nPerformance Metrics:")
    for k, v in mts.items():
        print(f'{k:22}: {v:,.4f}' if isinstance(v, float) else f'{k:22}: {v}')

    # Step 7: Plot results
    plot_results(df)
    return df, mts

if __name__ == '__main__':
    df_results, performance = main()