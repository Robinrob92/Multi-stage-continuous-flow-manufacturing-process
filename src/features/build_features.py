import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_pickle('../../data/interim/data_processed.pkl')




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

df['Stage1.Output.Measurement0.U.Actual']=ts_cleaned



def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate the average temperature across all machines
    df['Avg_MaterialTemperature'] = (
        df['Machine1.MaterialTemperature.U.Actual'] +
        df['Machine2.MaterialTemperature.U.Actual'] +
        df['Machine3.MaterialTemperature.U.Actual']
    ) / 3

    # Calculate the average motor amperage across all machines
    df['Avg_MotorAmperage'] = (
        df['Machine1.MotorAmperage.U.Actual'] +
        df['Machine2.MotorAmperage.U.Actual'] +
        df['Machine3.MotorAmperage.U.Actual']
    ) / 3

    # Calculate the average material pressure across all machines
    df['Avg_MaterialPressure'] = (
        df['Machine1.MaterialPressure.U.Actual'] +
        df['Machine2.MaterialPressure.U.Actual'] +
        df['Machine3.MaterialPressure.U.Actual']
    ) / 3

    # Calculate the average exit zone temperature across all machines
    df['Avg_ExitZoneTemperature'] = (
        df['Machine1.ExitZoneTemperature.C.Actual'] +
        df['Machine2.ExitZoneTemperature.C.Actual'] +
        df['Machine3.ExitZoneTemperature.C.Actual']
    ) / 3

    # Calculate the total raw material properties across all machines
    for i in range(1, 5):
        df[f'Total_RawMaterial_Property{i}'] = (
            df[f'Machine1.RawMaterial.Property{i}'] +
            df[f'Machine2.RawMaterial.Property{i}'] +
            df[f'Machine3.RawMaterial.Property{i}']
        )
    
    # Calculate the interaction between ambient conditions and average machine parameters
    df['AmbientHumidity_MaterialTemperature_Interaction'] = df['AmbientConditions.AmbientHumidity.U.Actual'] * df['Avg_MaterialTemperature']
    df['AmbientTemperature_MaterialTemperature_Interaction'] = df['AmbientConditions.AmbientTemperature.U.Actual'] * df['Avg_MaterialTemperature']
    df['AmbientHumidity_MotorAmperage_Interaction'] = df['AmbientConditions.AmbientHumidity.U.Actual'] * df['Avg_MotorAmperage']
    df['AmbientTemperature_MotorAmperage_Interaction'] = df['AmbientConditions.AmbientTemperature.U.Actual'] * df['Avg_MotorAmperage']
    
    return df

df_eng=add_engineered_features(df)

df_eng.shape

df_eng.to_pickle("../../data/interim/data_engineered.pkl")
