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
from pykalman import KalmanFilter

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

def apply_kalman_smart(series, window_size=500):
    """
    Impute missing values in a series using a windowed Kalman filter approach for each NaN.
    Args:
        series (pd.Series): The time series with possible NaNs.
        window_size (int): The size of the window to use around each missing value.
    Returns:
        pd.Series: The series with NaNs imputed where possible.
    """
    from pykalman import KalmanFilter
    series = series.copy()
    mask_nan = series.isna()
    filled_series = series.copy()
    missing_indices = series[mask_nan].index
    for idx in missing_indices:
        # Get window around the missing value
        start = max(series.index.get_loc(idx) - window_size // 2, 0)
        end = min(start + window_size, len(series))
        sub_index = series.index[start:end]
        sub_series = series.loc[sub_index]
        if sub_series.notna().sum() < 50:
            continue  # not enough data to fit Kalman
        # Pre-fill
        sub_series = sub_series.fillna(method='ffill').fillna(method='bfill')
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=sub_series.iloc[0],
            observation_covariance=1,
            transition_covariance=0.01
        )
        try:
            kf = kf.em(sub_series.values, n_iter=3)
            state_means, _ = kf.smooth(sub_series.values)
            smoothed_series = pd.Series(state_means.flatten(), index=sub_series.index)
            filled_series.loc[idx] = smoothed_series.loc[idx]
        except Exception:
            continue  # fallback to whatever is filled already
    return filled_series

# --- GARCH-based missing data handler (copied from z_score_strategy.py) ---
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


def kalman_filter_spread_with_innovations(df):
    """
    Apply a Kalman filter to estimate the dynamic relationship between Nifty and Bank Nifty IVs. Computes the innovation (residual) and its rolling standard deviation for use in trading signals.

    Args:
        df (pd.DataFrame): DataFrame with 'nifty' and 'banknifty' columns.

    Returns:
        pd.DataFrame: DataFrame with Kalman filter parameters, innovation, and innovation_std columns added.
    """
    y = df['banknifty'].values
    x = df['nifty'].values

    # Check for any remaining NaNs
    nan_mask = np.isnan(y) | np.isnan(x)
    if nan_mask.any():
        print(f"Warning: Found {nan_mask.sum()} NaN values in data, forward filling...")
        y_series = pd.Series(y, index=df.index).ffill().bfill()
        x_series = pd.Series(x, index=df.index).ffill().bfill()
        y = y_series.values
        x = x_series.values

    n = len(y)
    
    # Create observation matrix properly
    observation_matrix = np.zeros((n, 1, 2))
    observation_matrix[:, 0, 0] = x  # x coefficient (beta)
    observation_matrix[:, 0, 1] = 1  # intercept
    
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=observation_matrix,
        initial_state_mean=[1, 0],  # Start with beta=1, intercept=0
        initial_state_covariance=np.eye(2) * 100,  # Allow more initial uncertainty
        observation_covariance=1.0,
        transition_covariance=np.eye(2) * 0.01  # Increased for more adaptation
    )
    
    try:
        state_means, state_covs = kf.filter(y)
        beta, intercept = state_means[:, 0], state_means[:, 1]
        
        # Calculate predicted values and innovation
        predicted = beta * x + intercept
        innovation = y - predicted
        
        # Calculate innovation standard deviation from state covariances
        # Use a rolling window approach for more stable estimates
        window = 100
        innovation_std = np.zeros_like(innovation)
        
        for i in range(len(innovation)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            window_innovations = innovation[start_idx:end_idx]
            innovation_std[i] = np.std(window_innovations) if len(window_innovations) > 10 else 1.0
        
        # Ensure minimum std to avoid division by zero
        innovation_std = np.maximum(innovation_std, 0.001)
        
    except Exception as e:
        print(f"Kalman filter failed: {e}")
        # Fallback to simple linear regression
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(x.reshape(-1, 1), y)
        beta = np.full(n, reg.coef_[0])
        intercept = np.full(n, reg.intercept_)
        predicted = beta * x + intercept
        innovation = y - predicted
        innovation_std = np.full(n, np.std(innovation))
    
    df['kalman_beta'] = beta
    df['kalman_intercept'] = intercept
    df['spread'] = innovation
    df['innovation'] = innovation
    df['innovation_std'] = innovation_std
    
    # Add some debugging info
    print(f"Innovation statistics:")
    print(f"  Mean: {np.mean(innovation):.6f}")
    print(f"  Std: {np.std(innovation):.6f}")
    print(f"  Min: {np.min(innovation):.6f}")
    print(f"  Max: {np.max(innovation):.6f}")
    print(f"Innovation Std statistics:")
    print(f"  Mean: {np.mean(innovation_std):.6f}")
    print(f"  Min: {np.min(innovation_std):.6f}")
    print(f"  Max: {np.max(innovation_std):.6f}")
    
    return df


def generate_kalman_innovation_signals(df, entry_mult=1.5, exit_mult=0.5):
    """
    Generate trading signals and positions based on normalized innovation (innovation divided by its estimated standard deviation). Entry and exit signals are determined by comparing normalized innovation to specified thresholds.

    Args:
        df (pd.DataFrame): DataFrame with 'innovation' and 'innovation_std' columns.
        entry_mult (float, optional): Threshold multiplier for entering trades. Defaults to 1.5.
        exit_mult (float, optional): Threshold multiplier for exiting trades. Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame with position, signal, position_change, and normalized_innovation columns added.
    """
    # Ensure no NaNs in innovation data
    if df['innovation'].isna().any() or df['innovation_std'].isna().any():
        print("Warning: NaNs found in innovation data, filling...")
        df['innovation'] = df['innovation'].ffill().bfill()
        df['innovation_std'] = df['innovation_std'].ffill().bfill()
    # Calculate normalized innovation
    normalized_innovation = df['innovation'] / df['innovation_std']
    # Generate entry and exit signals
    entry_long = normalized_innovation < -entry_mult  # Innovation is significantly negative
    entry_short = normalized_innovation > entry_mult   # Innovation is significantly positive
    exit_signal = np.abs(normalized_innovation) < exit_mult  # Innovation close to zero
    # Generate raw position signals
    raw_signals = np.where(exit_signal, 0,
                  np.where(entry_long, 1,
                  np.where(entry_short, -1, np.nan)))
    # Forward fill to maintain positions
    df['position'] = pd.Series(raw_signals, index=df.index).ffill().fillna(0).astype(np.int8)
    # Force close positions at end of day
    eod_times = ['15:30:00', '15:29:00', '15:28:00']  # Close near end of day
    for eod_time in eod_times:
        eod_mask = df.index.time == pd.to_datetime(eod_time).time()
        df.loc[eod_mask, 'position'] = 0
    # Calculate position changes (signals)
    df['signal'] = df['position'].diff().fillna(df['position']).astype(np.int8)
    df['position_change'] = df['signal']
    # Add normalized_innovation for analysis
    df['normalized_innovation'] = normalized_innovation
    # Debug information
    print(f"Signal generation statistics:")
    print(f"  Entry long signals: {entry_long.sum()}")
    print(f"  Entry short signals: {entry_short.sum()}")
    print(f"  Exit signals: {exit_signal.sum()}")
    print(f"  Non-zero positions: {(df['position'] != 0).sum()}")
    print(f"  Position changes: {(df['signal'] != 0).sum()}")
    print(f"  Normalized innovation range: [{normalized_innovation.min():.2f}, {normalized_innovation.max():.2f}]")
    return df


def calculate_pnl(df, trade_cost_bps=0.0325):
    """
    Calculate trade-by-trade and cumulative profit and loss (P&L) for the strategy. P&L is computed for each trade based on innovation change, average TTE, trade direction, and transaction costs.

    Args:
        df (pd.DataFrame): DataFrame with 'position', 'spread', and 'tte' columns.
        trade_cost_bps (float, optional): Transaction cost per trade. Defaults to 0.0325.

    Returns:
        pd.DataFrame: DataFrame with 'trade_pnl' and 'cumulative_pnl' columns added.
    """
    df['trade_pnl'] = 0.0
    cumulative_pnl = 0.0

    position = 0
    entry_spread = None
    entry_tte = None
    
    print("Calculating P&L...")
    trades_executed = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        new_position = row['position']
        
        # Check if position changed
        if new_position != position:
            # If we had an open position, close it
            if position != 0 and entry_spread is not None:
                spread_change = row['spread'] - entry_spread
                avg_tte = ((entry_tte + row['tte']) / 2.0) ** 0.7
                pnl = spread_change * avg_tte * position - trade_cost_bps
                df.at[idx, 'trade_pnl'] = pnl
                cumulative_pnl += pnl
                trades_executed += 1
                
                if trades_executed <= 10:  # Debug first few trades
                    print(f"Trade {trades_executed}: pos={position}, spread_change={spread_change:.6f}, "
                          f"avg_tte={avg_tte:.2f}, pnl={pnl:.6f}")
            
            # Update position and entry details
            position = new_position
            if position != 0:
                entry_spread = row['spread']
                entry_tte = row['tte']
            else:
                entry_spread = None
                entry_tte = None

    df['cumulative_pnl'] = df['trade_pnl'].cumsum()
    print(f"Total trades executed: {trades_executed}")
    
    return df


def calculate_performance_metrics(df):
    """
    Compute key performance metrics for the strategy, including total P&L, number of trades, Sharpe ratio, drawdown, and win rate.

    Args:
        df (pd.DataFrame): DataFrame with 'cumulative_pnl' and 'trade_pnl' columns.

    Returns:
        dict: Dictionary of performance metrics.
    """
    if df['cumulative_pnl'].iloc[-1] == 0:
        print("Warning: No P&L generated, returning zero metrics")
        return {
            'Total P/L': 0,
            'Number of Trades': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown (abs)': 0,
            'Max Drawdown (%)': 0,
            'Win Rate': 0
        }
    
    total_pnl = df['cumulative_pnl'].iloc[-1]
    num_trades = (df['trade_pnl'] != 0).sum()

    # Daily P/L for Sharpe calculation
    daily_pnl = df['cumulative_pnl'].resample('D').last().diff().dropna()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() != 0 and len(daily_pnl) > 1 else 0)

    # Drawdown
    equity = df['cumulative_pnl']
    running_max = equity.cummax()
    drawdown = equity - running_max
    drawdown_pct = drawdown / running_max.replace(0, np.nan)
    max_dd = drawdown.min()
    max_dd_pct = drawdown_pct.min() if not drawdown_pct.isna().all() else 0

    # Win rate
    trade_pnls = df.loc[df['trade_pnl'] != 0, 'trade_pnl']
    win_rate = (trade_pnls > 0).mean() if len(trade_pnls) > 0 else 0

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
    Plot the main results of the strategy, including innovation, normalized innovation, positions, cumulative P&L, drawdown, and Kalman filter parameters. Saves the plot as a PNG in the output directory.

    Args:
        df (pd.DataFrame): DataFrame with strategy results and metrics columns.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    print(df.index.to_series().diff().value_counts().head(10))

    output_dir = 'Kalman-Strategy-Plots'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'kalman_strategy_results_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)

    fig, axes = plt.subplots(7, 1, figsize=(15, 28), sharex=True)

    # Plot 1: Innovation (Spread) with signals
    axes[0].plot(df.index, df['innovation'], label='Innovation', alpha=0.7, linewidth=0.8)
    
    # Add entry/exit thresholds
    entry_threshold = 1.5 * df['innovation_std']
    axes[0].plot(df.index, entry_threshold, '--', color='red', alpha=0.5, label='Entry Threshold (+)')
    axes[0].plot(df.index, -entry_threshold, '--', color='red', alpha=0.5, label='Entry Threshold (-)')
    
    # Mark trades
    trade_entries = df[df['signal'] != 0]
    if len(trade_entries) > 0:
        axes[0].scatter(trade_entries.index, trade_entries['innovation'], 
                       c=['green' if x > 0 else 'red' for x in trade_entries['signal']], 
                       s=30, alpha=0.8, label='Trade Signals', zorder=5)
    
    axes[0].set_ylabel('Innovation')
    axes[0].set_title('Kalman Innovation with Trading Signals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Normalized Innovation 
    if 'normalized_innovation' in df.columns:
        axes[1].plot(df.index, df['normalized_innovation'], label='Normalized Innovation', color='purple', alpha=0.7)
        axes[1].axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Entry Threshold')
        axes[1].axhline(y=-1.5, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Exit Threshold')
        axes[1].axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Normalized Innovation')
        axes[1].set_title('Normalized Innovation (Model-based Z-Score Equivalent)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Position
    axes[2].step(df.index, df['position'], label='Position', color='orange', where='post')
    axes[2].set_ylabel('Position')
    axes[2].set_title('Trading Position Over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Cumulative PnL
    axes[3].plot(df.index, df['cumulative_pnl'], label='Cumulative P/L', color='green', linewidth=1.5)
    axes[3].set_ylabel('Cumulative P/L')
    axes[3].set_title('Strategy Performance (Cumulative P&L)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Drawdown
    if df['cumulative_pnl'].max() > 0:
        running_max = df['cumulative_pnl'].cummax()
        drawdown = df['cumulative_pnl'] - running_max
        axes[4].fill_between(df.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        axes[4].set_ylabel('Drawdown')
        axes[4].set_title('Drawdown Over Time')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

    # Plot 6: Kalman Beta (dynamic hedge ratio)
    axes[5].plot(df.index, df['kalman_beta'], label='Kalman β (hedge ratio)', color='blue')
    axes[5].set_ylabel('β')
    axes[5].set_title('Kalman Filter – Dynamic Hedge Ratio')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    # Plot 7: Kalman Intercept
    axes[6].plot(df.index, df['kalman_intercept'], label='Kalman α (intercept)', color='purple')
    axes[6].set_ylabel('α')
    axes[6].set_title('Kalman Filter – Dynamic Intercept')
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filepath}")
    plt.show()


def main():
    """
    Run the full Kalman filter strategy pipeline: load data, preprocess, calculate signals, evaluate, and plot results. Prints performance metrics and saves plots.

    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame with results and dictionary of performance metrics.
    """
    try:
        # Step 1: Load and inspect data
        df = load_and_inspect_data('data.parquet')
        
        # Step 2: Handle missing data
        df = handle_missing_data_ffill_garch(df)
        print(f"After handling missing data: {df.shape[0]} rows")
        
        # Step 3: Kalman filter and innovation calculation
        df = kalman_filter_spread_with_innovations(df)
        
        # Step 4: Generate trading signals
        df = generate_kalman_innovation_signals(df, entry_mult=1.5, exit_mult=0.5)
        
        # Step 5: Calculate P&L
        df = calculate_pnl(df)
        
        # Step 6: Calculate performance metrics
        mts = calculate_performance_metrics(df)

        # Print performance metrics
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        for k, v in mts.items():
            if isinstance(v, float):
                print(f'{k:22}: {v:,.6f}')
            else:
                print(f'{k:22}: {v}')

        # Step 7: Plot results
        plot_results(df)
        
        return df, mts
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    df_results, performance = main()

