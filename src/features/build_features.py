import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('../../data/raw/continuous_factory_process.csv')

df['Stage1.Output.Measurement0.U.Actual'].plot()


def clean_timeseries(ts, threshold=3):
    """
    Removes extreme values from a time series and fills in missing values using linear interpolation.
    
    Args:
        ts (pd.Series): The time series to clean.
        threshold (int, optional): The number of standard deviations from the mean at which values are considered extreme.
            Defaults to 3.
    
    Returns:
        pd.Series: The cleaned time series.
    """
    
    # Calculate the mean and standard deviation of the time series
    ts_mean = ts.mean()
    ts_std = ts.std()
    
    # Remove extreme values
    ts[ts > ts_mean + threshold * ts_std] = np.nan
    ts[ts < ts_mean - threshold * ts_std] = np.nan
    
    # Fill in missing values with linear interpolation
    ts = ts.interpolate()
    
    return ts

# Clean the time series
ts_cleaned = clean_timeseries(df['Stage1.Output.Measurement0.U.Actual'])
ts_cleaned.plot()