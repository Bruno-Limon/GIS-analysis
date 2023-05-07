import numpy as np
import pandas as pd
import collections
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

"""DATA CLEANING"""

air_quality_index = 'https://raw.githubusercontent.com/Bruno-Limon/air-quality-analysis/main/Data/AQI-2016.csv'
df_aqi = pd.read_csv(air_quality_index)
df_aqi.insert(0, 'Datetime', pd.to_datetime(df_aqi['Fecha'] + ' ' + df_aqi['Hora']))
df_aqi.drop(['Fecha', 'Hora'], axis=1, inplace=True)

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
df_aqi.drop(['CASETA '], axis=1, inplace=True)

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

# Re-imputing the values with the new clean dataset
df_aqi = df_aqi.fillna(method="ffill")

"""OUTLIER DETECTION"""

# Fitting a PCA model with 2 components to visualize
X_train = df_aqi.iloc[:, 1:9].values
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# DBSCAN model using eps obtained from previous elbow method
dbscan = DBSCAN(eps=3.8, min_samples=4)
dbscan.fit(X_train_pca)
# Identifiying outliers found by DBSCAN
anomalies_db = np.where(dbscan.labels_ == -1)
anomalies_db_pca = X_train_pca[anomalies_db]
print('Outliers found by DBSCAN:', len(anomalies_db_pca))

# Isolation Forest model
isol = IsolationForest(bootstrap=True,
                       contamination=0.002,
                       max_samples=200,
                       n_estimators=1000,
                       n_jobs=-1).fit(X_train_pca)
# Identifiying outliers found by Isolation Forest
outliers_isol = isol.predict(X_train_pca)
anomalies_isol = np.where(outliers_isol == -1)
anomalies_isol_pca = X_train_pca[anomalies_isol]

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=1000,
                         contamination=.002,
                         algorithm='kd_tree')
# Identifiying outliers found by Local Outlier Factor
outliers_lof = lof.fit_predict(X_train_pca)
anomalies_lof = np.where(outliers_lof == -1)
anomalies_lof_pca = X_train_pca[anomalies_lof]

# Making list with all outliers, including duplicates
list_all_outliers = []
for i, j, k in zip(anomalies_db, anomalies_lof, anomalies_lof):
    list_all_outliers.extend(i)
    list_all_outliers.extend(j)
    list_all_outliers.extend(k)

# Selecting outliers that were obtained with at least 2/3 methods
list_final_outliers = []
for item, count in collections.Counter(list_all_outliers).items():
    if count >= 2:
        list_final_outliers.append(item)

# Creating dataframe without anoamlies
df_aqi_noanomalies = df_aqi.copy()
df_aqi_noanomalies.drop(list_final_outliers, inplace=True)
df_aqi_noanomalies = df_aqi_noanomalies.reset_index(drop=True)