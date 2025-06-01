from typing import Tuple
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import time
import datetime
import glob

import sympy as sp
from scipy.integrate import quad

from pandas.core.api import DataFrame

from kernel import KernelEstimator

def process_data(path_pattern: str) -> pd.DataFrame:
    # Initialize an empty list to store all results
    kernel = KernelEstimator()
    all_daily_vol_results = []

    for file_path in glob.glob(path_pattern):
        # Read the current CSV file
        print("processing", file_path)
        raw_data = pd.read_csv(file_path)
        raw_data = prepare_data(raw_data)
        clean_data, closing_prices = clean_taq(raw_data, k=4)
        full_data = generate_second_by_second_returns(clean_data)
        
        for (symbol, date), group in tqdm(full_data.groupby(['SYMBOL', 'DATE'])):

            naive = group.copy()
            naive['datetime'] = pd.to_datetime(naive['DATE'].astype(str) + ' ' + naive['TIME'].astype(str))
            naive = naive.set_index('datetime')
            sampled_prices = naive['PRICE'].resample('5min').last().dropna()
            sampled_log_returns = np.log(sampled_prices / sampled_prices.shift(1)).dropna()
            naive_daily_vol = sum(sampled_log_returns ** 2)

            # Extract log returns for this symbol-date combination
            log_returns = group['log_return'].dropna()
                
            # Apply the kernel's estimate_daily_vol method
            daily_vol = kernel.estimate_daily_vol(log_returns, kernel_type="truncated_2", optimize_bandwidth=True, optimize_theta=False)
            
            # Store the result
            all_daily_vol_results.append({
                'SYMBOL': symbol,
                'DATE': date,
                'daily_volatility': daily_vol,
                'naive_daily_volatility': naive_daily_vol,
                'close_price': closing_prices[
                    (closing_prices['SYMBOL'] == symbol) & 
                    (closing_prices['DATE'] == date)
                ]['PRICE'].iloc[0]
            })

    # Convert the list of dictionaries to a DataFrame at the end
    result_df = pd.DataFrame(all_daily_vol_results)
    result_df = result_df.sort_values(by=['SYMBOL', 'DATE'])
    return result_df

def remove_outliers_with_sliding_window_mad(df, window_size='6min', slide_step='2min', 
                                           price_col='PRICE', k=5, min_windows_flagged=2):
    """
    Detect outliers using MAD with overlapping time windows
    
    Parameters:
    -----------
    df : pandas DataFrame
        The data to clean
    window_size : str
        Size of each window (e.g., '1h' for 1 hour) - note use 'h' not 'H'
    slide_step : str
        How much to slide the window (e.g., '30min' for 30 minutes)
    price_col : str
        Column name containing price data
    k : float
        Threshold multiplier
    min_windows_flagged : int
        Minimum number of windows that must flag a point as outlier
        
    Returns:
    --------
    DataFrame without outliers
    """
    # Make a copy of the input dataframe
    result_df = df.copy()
    
    # Initialize outlier counter column - using integer instead of boolean
    result_df['outlier_count'] = 0
    
    # Group by symbol and date
    for (symbol, date), group in tqdm(result_df.groupby(['SYMBOL', 'DATE']), desc="Removing outliers..."):
        # Convert TIME column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(group['TIME']):
            time_col = pd.to_datetime(group['TIME'], format='%H:%M:%S')
        else:
            time_col = group['TIME']
        
        # Get start and end times for this group
        start_time = time_col.min()
        end_time = time_col.max()
        
        # Calculate the duration and make sure we have valid windows
        duration = end_time - start_time
        window_delta = pd.Timedelta(window_size)
        
        # Skip if duration is too short for a full window
        if duration < window_delta:
            continue
            
        # Calculate the last valid window start time
        last_window_start = start_time + duration - window_delta
        
        # Generate window start times
        window_starts = pd.date_range(start=start_time, 
                                      end=last_window_start, 
                                      freq=slide_step)
        
        # For each window
        for start in window_starts:
            end = start + pd.Timedelta(window_size)
            # Filter data within the time window
            window_mask = (time_col >= start) & (time_col <= end)
            window_data = group[window_mask]
            
            if len(window_data) < 3:  # Skip windows with too few points
                continue
            
            # Calculate MAD
            median_val = window_data[price_col].median()
            mad = np.median(np.abs(window_data[price_col] - median_val))
            
            # Skip if MAD is 0 (all values identical)
            if mad == 0:
                continue
                
            # Define bounds
            lower_bound = median_val - (k * mad * 1.4826)
            upper_bound = median_val + (k * mad * 1.4826)
            
            # Identify outliers in this window
            outliers_in_window = (window_data[price_col] < lower_bound) | (window_data[price_col] > upper_bound)
            
            # Increment counter for outlier points
            outlier_indices = window_data.index[outliers_in_window]
            result_df.loc[outlier_indices, 'outlier_count'] += 1
    
    # Filter out rows that were flagged as outliers in at least min_windows_flagged windows
    return result_df[result_df['outlier_count'] < min_windows_flagged]

def clean_taq(taq_raw:DataFrame, k) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Removes incorrect and delayed trades
    taq_filtered = taq_raw[(taq_raw['CORR'] == 0) & (taq_raw['COND'] != 'Z')].copy()

    # Group by symbol, date, time and take median price and sum of size
    second_level_data = taq_filtered.groupby(['SYMBOL', 'DATE', 'TIME']).agg({
        'PRICE': 'median',  # Median price from all trades in this second
        #'SIZE': 'sum',      # Total volume in this second
    }).reset_index()

    # Create time bounds
    market_open = time(9, 30, 0)   # 9:30 AM
    market_close = time(16, 5, 0)  # 4:05 PM
    # Filter for trades within market hours
    almost_clean = second_level_data[
        (second_level_data['TIME'] >= market_open) & 
        (second_level_data['TIME'] <= market_close)
    ]

    with_outliers = len(almost_clean)
    clean = remove_outliers_with_sliding_window_mad(almost_clean, k=k)
    without_outliers = len(clean)

    clean_sorted = clean.sort_values(['SYMBOL', 'DATE', 'TIME'])

    closing_prices = clean_sorted.groupby(['SYMBOL', 'DATE']).last().reset_index()
    closing_prices.drop(['TIME', 'outlier_count'], axis=1, inplace=True)

    p_outliers = 100 * round(np.abs((without_outliers / with_outliers) - 1), 4)
    print(f"Percentage of data removed as outliers: {p_outliers}%")

    analysis_start = time(9, 35, 0)
    analysis_end = time(15, 55, 0)

    final = clean[
        (clean['TIME'] >= analysis_start) & 
        (clean['TIME'] <= analysis_end)
    ]

    return final, closing_prices

def naive_daily_volatility(df, date_col='DATE', time_col='TIME', price_col='PRICE', symbol_col='SYMBOL', freq='5min'):
    """
    Calculate daily realized volatility using 5-minute sampled returns
    
    Parameters:
    -----------
    df : pandas DataFrame
        The cleaned data with prices
    date_col : str
        Column containing the date
    time_col : str
        Column containing the time
    price_col : str
        Column containing the price
    symbol_col : str
        Column containing the symbol identifier
    freq : str
        Sampling frequency (default '5min')
        
    Returns:
    --------
    DataFrame with daily volatility measures for each symbol
    """
    # Sort by symbol and datetime
    df = df.sort_values([symbol_col, 'DATE', 'TIME'])
    
    # Group by symbol and date
    result = []
    
    for (symbol, date), group in tqdm(df.groupby([symbol_col, date_col]), desc="Calculating naive daily integrated variance..."):
        if 'datetime' not in group.columns:
            # If DATE and TIME are separate columns, combine them
            group['datetime'] = pd.to_datetime(group['DATE'].astype(str) + ' ' + group['TIME'].astype(str))
        
        # Set datetime as index for resampling
        group = group.set_index('datetime')

        # Resample to 5-minute intervals, taking the last price in each interval
        sampled = group[price_col].resample(freq).last().dropna()
        
        # Need at least 2 observations to compute returns
        if len(sampled) < 2:
            continue
            
        # Calculate log returns
        log_returns = np.log(sampled / sampled.shift(1)).dropna()
        
        # Square returns
        squared_returns = log_returns ** 2
        
        # Calculate realized volatility (sum of squared returns)
        realized_vol = np.sqrt(squared_returns.sum())
        
        result.append({
            'symbol': symbol,
            'date': date,
            'n_observations': len(sampled),
            'realized_volatility': realized_vol,
        })
    
    return pd.DataFrame(result)

def prepare_data(df):
    """
    Prepare the DataFrame by ensuring correct data types
    """
    # Make sure DATE is a datetime.date object
    if not isinstance(df['DATE'].iloc[0], datetime.date):
        df['DATE'] = pd.to_datetime(df['DATE']).dt.date

    if 'TIME_M' in df.columns:
        # Convert to datetime with the correct format (note %f for microseconds, not %M for month)
        df['TIME'] = pd.to_datetime(df['TIME_M'], format='%H:%M:%S.%f')
        # Extract just the time part (up to seconds)
        df['TIME'] = df['TIME'].dt.strftime('%H:%M:%S')
        # Drop the original column with milliseconds
        df = df.drop('TIME_M', axis=1)
    
    # Make sure TIME is a datetime.time object
    if 'TIME' in df.columns and not isinstance(df['TIME'].iloc[0], datetime.time):
        # Assuming TIME is in format 'HH:MM:SS'
        df['TIME'] = pd.to_datetime(df['TIME'], format='%H:%M:%S').dt.time

    df.columns = ['SYMBOL' if x == 'SYM_ROOT' else x for x in df.columns]
    df.columns = ['COND' if x == 'TR_SCOND' else x for x in df.columns]
    df.columns = ['CORR' if x == 'TR_CORR' else x for x in df.columns]

    if 'SYM_SUFFIX' in df.columns:
        df = df.drop('SYM_SUFFIX', axis=1)
    
    return df

def generate_second_by_second_returns(df):
    """
    Generate second-by-second data with log returns
    """
    # Compute log returns for observed data points
    df = df.sort_values(['SYMBOL', 'DATE', 'TIME'])
    df['log_return'] = df.groupby(['SYMBOL', 'DATE'])['PRICE'].transform(
        lambda x: np.log(x / x.shift(1))
    )

    result_dfs = []
    
    for (date, symbol), group_df in df.groupby(['DATE', 'SYMBOL']):
        # Get min and max time for this group
        min_time = group_df['TIME'].min()
        max_time = group_df['TIME'].max()
        
        # Convert to seconds for easy iteration
        min_seconds = min_time.hour * 3600 + min_time.minute * 60 + min_time.second
        max_seconds = max_time.hour * 3600 + max_time.minute * 60 + max_time.second
        
        # Create all seconds in the range
        all_times = []
        for sec in range(min_seconds, max_seconds + 1):
            hour = sec // 3600
            minute = (sec % 3600) // 60
            second = sec % 60
            all_times.append(datetime.time(hour, minute, second))
        
        # Create a DataFrame with all seconds
        all_seconds_df = pd.DataFrame({
            'DATE': date,
            'SYMBOL': symbol,
            'TIME': all_times
        })
        
        # Merge with observed data
        merged_df = pd.merge(
            all_seconds_df,
            group_df[['DATE', 'SYMBOL', 'TIME', 'PRICE', 'log_return']],
            on=['DATE', 'SYMBOL', 'TIME'],
            how='left'
        )
        
        # Fill missing log returns with 0
        merged_df['log_return'] = merged_df['log_return'].fillna(0)
        
        result_dfs.append(merged_df)
    
    return pd.concat(result_dfs).sort_values(['DATE', 'SYMBOL', 'TIME'])


if __name__ == "__main__":
    path_pattern = './data/raw/ibm_*.csv'
    output_path = './data/output/ibm_realized_vol_prc_2000-2024.csv'

    df = process_data(path_pattern)

    # Save the merged DataFrame to a CSV file
    df.to_csv(output_path, index=False)

