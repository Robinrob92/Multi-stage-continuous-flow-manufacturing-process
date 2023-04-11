import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('../../data/raw/continuous_factory_process.csv')
df.head()

df.info(verbose=True)




def process_dataframe(df):
    """
    This function takes a DataFrame as input and performs the following operations:
    - Select the first 71 columns
    - Remove columns that contain "Setpoint" in their name
    - Convert the 'time_stamp' column to a datetime data type
    - Set the 'time_stamp' column as the index
    
    Args:
    - df (pandas.DataFrame): The input DataFrame to be processed
    
    Returns:
    - pandas.DataFrame: A processed DataFrame with the first 71 columns, columns containing "Setpoint" removed,
                        the 'time_stamp' column converted to a datetime data type, and the 'time_stamp' column
                        set as the index.
    """
    # Select the first 71 columns
    selected_columns = df.columns[:71]
    df = df[selected_columns]
    
    # Remove columns that contain "Setpoint" in their name
    df = df.loc[:, ~df.columns.str.contains('Setpoint')]
    
    # Convert the 'time_stamp' column to a datetime data type
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    
    # Set the 'time_stamp' column as the index
    df = df.set_index('time_stamp')
    
    return df

processed_df = process_dataframe(df)

processed_df.head()

processed_df.to_pickle("../../data/interim/data_processed.pkl")