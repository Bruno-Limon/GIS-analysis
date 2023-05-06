import numpy as np
import pandas as pd

air_quality_index = 'https://raw.githubusercontent.com/Bruno-Limon/air-quality-analysis/main/Data/AQI-2016.csv'
df_aqi = pd.read_csv(air_quality_index)
df_aqi.info(verbose = 1)
print(df_aqi)

df_aqi.insert(0, 'Datetime', pd.to_datetime(df_aqi['Fecha'] + ' ' + df_aqi['Hora']))
# df_aqi.insert(0, 'Datetime', pd.to_datetime(df_aqi['Fecha'] + ' ' + df_aqi['Hora'], format = 'mixed'))
df_aqi.drop(['Fecha', 'Hora'], axis = 1, inplace = True)

# Looking at unique values and time span
print('Distinct monitoring stations:', df_aqi['CASETA '].unique())
print('Earliest date:', df_aqi['Datetime'].min())
print('Latest date:', df_aqi['Datetime'].max())

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

list_monitor_station = list(df_aqi['Station'].unique())
list_columns = list(df_aqi.columns.values)[1:9]

# Creating list containing null values filtered by the monitoring station
list_nulls = []
for i, station in enumerate(list_monitor_station):
    list_nulls.append([])
    for j in range(len(list_columns)):
        list_nulls[i].append(
            df_aqi.loc[df_aqi['Station'] == station].iloc[:, j+1].isnull().sum())

# Putting the lists into a dataframe
df_aqi_nulls = pd.DataFrame(np.column_stack([list_nulls[i] for i in range(len(list_monitor_station))]),
                            columns=list_monitor_station,
                            index=list_columns)
print(df_aqi_nulls.T)
