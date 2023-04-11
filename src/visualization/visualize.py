import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('../../data/raw/continuous_factory_process.csv')

df.info(verbose=True)

import matplotlib.pyplot as plt


for  col in df.columns:
    print(col)

import matplotlib.pyplot as plt



def plot_feature(data, feature_name):
    plt.figure(figsize=(10, 6))
    plt.plot(data[f'Machine1.{feature_name}'], label='Machine 1', alpha=0.7)
    plt.plot(data[f'Machine2.{feature_name}'], label='Machine 2', alpha=0.7)
    plt.plot(data[f'Machine3.{feature_name}'], label='Machine 3', alpha=0.7)
    plt.legend()
    plt.title(f'{feature_name} for Machines 1, 2, and 3')
    plt.xlabel('Sample Index')
    plt.ylabel(f'{feature_name}')
    plt.show()

feature_list = [
    'RawMaterial.Property1',
    'RawMaterial.Property2',
    'RawMaterial.Property3',
    'RawMaterial.Property4',
    'RawMaterialFeederParameter.U.Actual',
    'Zone1Temperature.C.Actual',
    'Zone2Temperature.C.Actual',
    'MotorAmperage.U.Actual',
    'MotorRPM.C.Actual',
    'MaterialPressure.U.Actual',
    'MaterialTemperature.U.Actual',
    'ExitZoneTemperature.C.Actual'
]

for feature in feature_list:
    plot_feature(df, feature)
