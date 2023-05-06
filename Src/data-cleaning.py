import numpy as np
import pandas as pd

air_quality_index = 'https://raw.githubusercontent.com/Bruno-Limon/air-quality-analysis/main/Data/AQI-2016.csv'
df_aqi = pd.read_csv(air_quality_index)
df_aqi.insert(0, 'Datetime', pd.to_datetime(df_aqi['Fecha'] + ' ' + df_aqi['Hora']))
df_aqi.drop(['Fecha', 'Hora'], axis = 1, inplace = True)

# Mapping the 3 letter code of monitoring stations to their full name, for clarity
dict_caseta = {'ATM': 'Atemajac',
               'OBL': 'Oblatos',
               'PIN': 'Las Pintas',
               'SFE': 'Santa Fe',
               'TLA': 'Tlaquepaque',
               'VAL': 'Vallarta',
               'CEN': 'Centro',
               'AGU': 'Las Aguilas',
               'LDO': 'Loma Dorada',
               'MIR': 'Miravalle', }

df_aqi["Caseta"] = df_aqi['CASETA '].map(dict_caseta)
df_aqi.drop(['CASETA '], axis = 1, inplace = True)

# Dropping unncessary features or those with over 50% missing values
to_drop = ['PM2.5 ', 'Radiacion solar', 'Indice UV', 'Direccion de vientos ']
df_aqi.drop(to_drop, axis=1, inplace=True)
df_aqi.rename(columns={'Humedad relativa ': 'Humidity',
                       'Velocidad de viento ': 'Wind velocity',
                       'Caseta': 'Station',
                       'Temperatura': 'Temperature'}, inplace=True)

list_monitor_station = list(df_aqi['Station'].unique())
list_columns = list(df_aqi.columns.values)[1:9]

# Removing the monitoring stations with features consisting entirely of missing values as well as the anomaly in SO2
df_aqi = df_aqi[df_aqi.Station.str.contains('Santa Fe|Centro') == False]
df_aqi = df_aqi.drop([43816, 14554])