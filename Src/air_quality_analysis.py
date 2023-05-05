# %%
# General libraries
import numpy as np
import random
import pandas as pd
import datetime
import collections
from sklearn.impute import SimpleImputer
from scipy.stats import gaussian_kde, iqr

# Data visualization
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker as ticker
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

# Geospatial analysis
from geovoronoi import voronoi_regions_from_coords
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
import json
from scipy.spatial import Voronoi
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import unary_union
import osmnx as ox

# Outlier Detection
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest

# Dimensionality Reduction
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, chi2
from sklearn.decomposition import PCA

# Regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor

# Time Series Analysis
from scipy.spatial.distance import euclidean

# import skmob
# from skmob.tessellation import tilers
# from skmob.models.epr import DensityEPR, SpatialEPR, Ditras
# from skmob.models.markov_diary_generator import MarkovDiaryGenerator
# from skmob.preprocessing import filtering, compression, detection, clustering
# from skmob.measures.individual import jump_lengths, radius_of_gyration, uncorrelated_entropy, number_of_locations, number_of_visits, location_frequency
# from skmob.measures.collective import visits_per_location
# from skmob.utils.plot import *
# from skmob.data.load import load_dataset, list_datasets

# import sklearn
# from sklearn.metrics import mean_squared_error

# %%
# Setting up plot style
sns.set_context(font_scale=2, rc={"font.size": 10,
                                  "axes.titlesize": 16,
                                  "axes.labelsize": 14})
sns.set_style("whitegrid", {'grid.linestyle': '--', 'alpha': 0.25})
sns.set_style({'font.family': 'serif', 'font.serif': 'Computer Modern'})


# %% [markdown]
# # **<font color="#ffb94f">1.0 DATA UNDERSTANDING & PREPARATION</font>**

# %% [markdown]
# ## **<font color="#84f745">1.1 DATA UNDERSTANDING</font>**

# %%
# Taking a first look at the dataset
air_quality_index = 'https://raw.githubusercontent.com/Bruno-Limon/air-quality-analysis/main/Data/AQI-2016.csv'
df_aqi = pd.read_csv(air_quality_index)

display(pd.concat([df_aqi.head(2), df_aqi.tail(2)]))

# %%
df_aqi.info()


# %%
# Creating datetime column using date and hour columns
df_aqi.insert(0, 'Datetime', pd.to_datetime(
    df_aqi['Fecha'] + ' ' + df_aqi['Hora']))
# df_aqi.insert(0, 'Datetime', pd.to_datetime(df_aqi['Fecha'] + ' ' + df_aqi['Hora'], format = 'mixed'))
df_aqi.drop(['Fecha', 'Hora'], axis=1, inplace=True)

# Looking at unique values and time span
print('Distinct monitoring stations:', df_aqi['CASETA '].unique())
print('Earliest date:', df_aqi['Datetime'].min())
print('Latest date:', df_aqi['Datetime'].max())


# %%
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


# %% [markdown]
# MISSING DATA

# %%
# Looking at the amount of null values across features
print(df_aqi.isnull().sum(axis=0))

# Dropping unncessary features or those with over 50% missing values
to_drop = ['PM2.5 ', 'Radiacion solar', 'Indice UV', 'Direccion de vientos ']
df_aqi.drop(to_drop, axis=1, inplace=True)
df_aqi.rename(columns={'Humedad relativa ': 'Humidity',
                       'Velocidad de viento ': 'Wind velocity',
                       'Caseta': 'Station',
                       'Temperatura': 'Temperature'}, inplace=True)


# %%
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
df_aqi_nulls.T


# %% [markdown]
# As seen from the previous dataframe, some monitoring stations must have had some sensors not functioning thruoughout the year, such as temperature and humidity in the "Santa Fe" station, since every value concerning them is missing.
#
# This might make these stations useless for some of the analysis and algorithms applied to the data

# %%
# Dealing with missing values with different methods, imputing with mean, median, forward filling and dropping them altogether
df_aqi_dropna = df_aqi.dropna()
df_aqi_imputed1 = df_aqi.fillna(df_aqi.iloc[:, 1:9].mean())
df_aqi_imputed2 = df_aqi.fillna(df_aqi.iloc[:, 1:9].median())
df_aqi_imputed3 = df_aqi.fillna(method="ffill")

list_values_dropna = [df_aqi_dropna.iloc[:, i].values for i in range(1, 9)]
list_imputed_values1 = [df_aqi_imputed1.iloc[:, i].values for i in range(1, 9)]
list_imputed_values2 = [df_aqi_imputed2.iloc[:, i].values for i in range(1, 9)]
list_imputed_values3 = [df_aqi_imputed3.iloc[:, i].values for i in range(1, 9)]

# %%
# Obtaining distribution of each feature using KDE method
list_x = []
list_y = []
list_x1 = []
list_y1 = []
list_x2 = []
list_y2 = []
list_x3 = []
list_y3 = []

for i in range(len(list_values_dropna)):
    kde = gaussian_kde(list_values_dropna[i])
    x = np.linspace(min(list_values_dropna[i]), max(
        list_values_dropna[i]), 100)
    list_x.append(x)
    list_y.append(kde(x))

    kde1 = gaussian_kde(list_imputed_values1[i])
    x1 = np.linspace(min(list_imputed_values1[i]), max(
        list_imputed_values1[i]), 100)
    list_x1.append(x1)
    list_y1.append(kde1(x1))

    kde2 = gaussian_kde(list_imputed_values2[i])
    x2 = np.linspace(min(list_imputed_values2[i]), max(
        list_imputed_values2[i]), 100)
    list_x2.append(x2)
    list_y2.append(kde2(x2))

    kde3 = gaussian_kde(list_imputed_values3[i])
    x3 = np.linspace(min(list_imputed_values3[i]), max(
        list_imputed_values3[i]), 100)
    list_x3.append(x3)
    list_y3.append(kde3(x3))

# Plotting histogram for each feature in dataset
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 9))
k = 0
for i in range(2):
    for j in range(4):
        ax[i, j].hist(list_values_dropna[k], bins='scott', density=True)
        ax[i, j].plot(list_x[k], list_y[k], linewidth=3, alpha=.8)
        ax[i, j].set(title=list_columns[k])
        fig.tight_layout()
        k += 1


# %%
# Plotting distributions
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 9))
k = 0
for i in range(2):
    for j in range(4):
        ax[i, j].plot(list_x[k], list_y[k], label='dropna')
        ax[i, j].plot(list_x1[k], list_y1[k], label='fillna(mean)')
        ax[i, j].plot(list_x2[k], list_y2[k], label='fillna(median)')
        ax[i, j].plot(list_x3[k], list_y3[k], label='forward fill')
        ax[i, j].legend(loc='best')
        ax[i, j].set(title=list_columns[k])
        ax[i, j].set(ylabel="$P({})$".format(list_columns[k]))
        fig.tight_layout()
        k += 1


# %% [markdown]
# Looking at the comparison of different methods to deal with missing values, forward filling seems to be closer to the distribution of the values omitting NA, such is the case for the 1st 4 features for example, then, when looking at temperature and humidity, we get confirmation of the undesirable effects that imputing a large amount of values can heavily bias the distribution, by adding synthetic density into the mean areas of the distribution

# %% [markdown]
# Another interesting thing to note is that the distribution of SO2 went from a range of 0-0.016 when omitting nan values to 0-20 when imputing said values, given the orders of magnitude of this change, it might prove to be an anomaly

# %%
# Plotting histogram for each feature in dataset of values imputed with the mean to look deeper into the SO2 anomaly
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 9))
k = 0
for i in range(2):
    for j in range(4):
        ax[i, j].boxplot(list_imputed_values1[k])
        ax[i, j].set(title=list_columns[k])
        fig.tight_layout()
        k += 1


# %%
# Intuition was correct, SO2 contains a single outlier that completely skews the distribution
# To get a confirmation, we look at the 5 largest values of SO2
print('Top 5 SO2 values\n', df_aqi['SO2'].nlargest(5), '\n', sep="")
print('Top 5 O3 values\n', df_aqi['O3'].nlargest(5), sep="")


# %%
# Removing the monitoring stations with features consisting entirely of missing values as well as the anomaly in SO2
df_aqi_clean = df_aqi[df_aqi.Station.str.contains('Santa Fe|Centro') == False]
df_aqi_clean = df_aqi_clean.drop([43816, 14554])

# Re-imputing the values with the new clean dataset
df_aqi_clean_dropna = df_aqi_clean.dropna()
df_aqi_clean_imputed1 = df_aqi_clean.fillna(df_aqi_clean.iloc[:, 1:9].mean())
df_aqi_clean_imputed2 = df_aqi_clean.fillna(df_aqi_clean.iloc[:, 1:9].median())
df_aqi_clean_imputed3 = df_aqi_clean.fillna(method="ffill")

list_clean_values_dropna = [
    df_aqi_clean_dropna.iloc[:, i].values for i in range(1, 9)]
list_clean_imputed_values1 = [
    df_aqi_clean_imputed1.iloc[:, i].values for i in range(1, 9)]
list_clean_imputed_values2 = [
    df_aqi_clean_imputed2.iloc[:, i].values for i in range(1, 9)]
list_clean_imputed_values3 = [
    df_aqi_clean_imputed3.iloc[:, i].values for i in range(1, 9)]

# %%
# Repeating the distribution comparison to see the effects of the recent cleaning
list_clean_x = []
list_clean_y = []
list_clean_x1 = []
list_clean_y1 = []
list_clean_x2 = []
list_clean_y2 = []
list_clean_x3 = []
list_clean_y3 = []

for i in range(len(list_clean_values_dropna)):
    kde = gaussian_kde(list_clean_values_dropna[i])
    x = np.linspace(min(list_clean_values_dropna[i]), max(
        list_clean_values_dropna[i]), 100)
    list_clean_x.append(x)
    list_clean_y.append(kde(x))

    kde1 = gaussian_kde(list_clean_imputed_values1[i])
    x1 = np.linspace(min(list_clean_imputed_values1[i]), max(
        list_clean_imputed_values1[i]), 100)
    list_clean_x1.append(x1)
    list_clean_y1.append(kde1(x1))

    kde2 = gaussian_kde(list_clean_imputed_values2[i])
    x2 = np.linspace(min(list_clean_imputed_values2[i]), max(
        list_clean_imputed_values2[i]), 100)
    list_clean_x2.append(x2)
    list_clean_y2.append(kde2(x2))

    kde3 = gaussian_kde(list_clean_imputed_values3[i])
    x3 = np.linspace(min(list_clean_imputed_values3[i]), max(
        list_clean_imputed_values3[i]), 100)
    list_clean_x3.append(x3)
    list_clean_y3.append(kde3(x3))

# Plotting distributions
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 9))
k = 0
for i in range(2):
    for j in range(4):
        ax[i, j].plot(list_clean_x[k],  list_clean_y[k], label='dropna')
        ax[i, j].plot(list_clean_x1[k], list_clean_y1[k], label='fillna(mean)')
        ax[i, j].plot(list_clean_x2[k], list_clean_y2[k],
                      label='fillna(median)')
        ax[i, j].plot(list_clean_x3[k], list_clean_y3[k], label='forward fill')
        ax[i, j].legend(loc='best')
        ax[i, j].set(title=list_columns[k])
        ax[i, j].set(ylabel="$P({})$".format(list_columns[k]))
        fig.tight_layout()
        k += 1


# %% [markdown]
# The peaks in the temperature and humidity distributions have disappeared and the SO2 distribution got rid of the outlier, now it is clear that the dataset produced with forward filling imputing might be the best candidate to capture the underlying data distribution while maintining a decent amount of data

# %%
# Looking at stat summary of chosen dataset
df_aqi_clean_imputed3.iloc[:, 1:9].describe()

# %%
# Creating correlation matrix with pandas built-in methods
aqi_corr = df_aqi_clean_imputed3.iloc[:, 1:9].corr()
aqi_corr.style.background_gradient(cmap='RdBu', axis=None).set_precision(2)

# %% [markdown]
# ## **<font color="#84f745">1.2 OUTLIER DETECTION</font>**

# %%
# Fitting a PCA model with 2 components to visualize
X_train = df_aqi_clean_imputed3.iloc[:, 1:9].values
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Calculating distances among pair of neighbor points
nbr = NearestNeighbors(n_neighbors=2)
nbrs = nbr.fit(X_train_pca)
distances, indices = nbrs.kneighbors(X_train_pca)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]

# Plotting k-distances
fig = px.line(y=distances, title='K-distance elbow method')
fig.update_layout(xaxis_range=[0, 71000])
fig.show()


# %%
# DBSCAN model using eps obtained from previous elbow method
dbscan = DBSCAN(eps=3.8, min_samples=4)
dbscan.fit(X_train_pca)

# Identifiying outliers found by DBSCAN
anomalies_db = np.where(dbscan.labels_ == -1)
anomalies_db_pca = X_train_pca[anomalies_db]
print('Outliers found by DBSCAN:', len(anomalies_db_pca))

# Plotting outliers on the 2 dimensional PCA space
fig = plt.figure(figsize=(12, 4))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='skyblue', s=2)
plt.scatter(anomalies_db_pca[:, 0],
            anomalies_db_pca[:, 1], color='orangered', s=2)
plt.title("Outliers found by DBSCAN")
plt.show()


# %%
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
print('Outliers found by Isolation Forest:', len(anomalies_isol_pca))

# Plotting outliers
fig = plt.figure(figsize=(12, 4))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='skyblue', s=2)
plt.scatter(anomalies_isol_pca[:, 0],
            anomalies_isol_pca[:, 1], color='orangered', s=2)
plt.title("Outliers found by Isolation Forest")
plt.show()


# %%
# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=1000,
                         contamination=.002,
                         algorithm='kd_tree')

# Identifiying outliers found by Local Outlier Factor
outliers_lof = lof.fit_predict(X_train_pca)
anomalies_lof = np.where(outliers_lof == -1)
anomalies_lof_pca = X_train_pca[anomalies_lof]
print('Outliers found by LOF:', len(anomalies_lof_pca))

# Plotting outliers
fig = plt.figure(figsize=(12, 4))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='skyblue', s=2)
plt.scatter(anomalies_lof_pca[:, 0],
            anomalies_lof_pca[:, 1], color='orangered', s=2)
plt.title("Outliers found by Local Outlier Factor")
plt.show()


# %%
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

anomalies_final_pca = X_train_pca[list_final_outliers]
print('Final outliers:', len(anomalies_final_pca))

# Final plot of outliers
fig = plt.figure(figsize=(12, 4))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='skyblue', s=2)
plt.scatter(anomalies_final_pca[:, 0],
            anomalies_final_pca[:, 1], color='orangered', s=2)
plt.title("Outliers found by at least 2/3 methods")
plt.show()


# %%
# Creating dataframe without anoamlies
df_aqi_noanomalies = df_aqi_clean_imputed3.copy()
df_aqi_noanomalies.drop(list_final_outliers, inplace=True)
df_aqi_noanomalies = df_aqi_noanomalies.reset_index(drop=True)
X_train_noanomalies = df_aqi_noanomalies.iloc[:, 1:9].values

# Fitting PCA model with values without outliers
pca = PCA(n_components=2)
X_train_noanomalies_pca = pca.fit_transform(X_train_noanomalies)

# PCA 2 dimensional plot without outliers
fig = plt.figure(figsize=(12, 4))
plt.scatter(X_train_noanomalies_pca[:, 0],
            X_train_noanomalies_pca[:, 1], color='skyblue', s=2)
plt.scatter(anomalies_final_pca[:, 0],
            anomalies_final_pca[:, 1], color='white', s=2)
plt.title("PCA without outliers")
plt.show()


# %%
# Taking a look at the features after removing outliers
df_aqi_noanomalies.iloc[:, 1:9].describe()


# %%
# Plotting boxplot for each feature
list_values_noanomalies = [
    df_aqi_noanomalies.iloc[:, i].values for i in range(1, 9)]

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 9))
k = 0
for i in range(2):
    for j in range(4):
        ax[i, j].boxplot(list_values_noanomalies[k])
        ax[i, j].set(title=list_columns[k])
        fig.tight_layout()
        k += 1


# %%
# Obtaining distribution of each feature using KDE method
list_x = []
list_y = []

for i in range(len(list_values_noanomalies)):
    kde = gaussian_kde(list_values_noanomalies[i])
    x = np.linspace(min(list_values_noanomalies[i]), max(
        list_values_noanomalies[i]), 50)
    list_x.append(x)
    list_y.append(kde(x))

# Plotting histogram for each feature in dataset
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 9))
k = 0
for i in range(2):
    for j in range(4):
        ax[i, j].plot(list_x[k], list_y[k], linewidth=2)
        ax[i, j].set(title=list_columns[k])
        fig.tight_layout()
        k += 1


# %% [markdown]
# ## **<font color="#84f745">1.3 PREPARING DATA</font>**

# %%
# Assigning the new df_aqi dataset to the one without outliers and creating its training dataset
X_train = df_aqi_noanomalies.iloc[:, 1:9].values


# %% [markdown]
# PREPARING DATA IN DIFFERENT GRANULARITIES

# %%
# Grouping observations by individual days and monitoring point to end up with one record daily for each stations
df_aqi_daily_station = df_aqi_noanomalies.groupby(
    [df_aqi_noanomalies['Datetime'].dt.date, 'Station']).mean(numeric_only=False)
df_aqi_daily_station.drop(['Datetime'], axis=1, inplace=True)
df_aqi_daily_station = df_aqi_daily_station.reset_index()

df_aqi_daily_station.rename(columns={'Datetime': 'Date'}, inplace=True)
print('Number of records:', len(df_aqi_daily_station))
display(pd.concat([df_aqi_daily_station.head(
    2), df_aqi_daily_station.tail(2)]))


# %%
# Grouping observations by individual days and monitoring point to end up with one record daily for each stations
df_aqi_daily = df_aqi_noanomalies.iloc[:, 1:9].groupby(
    [df_aqi_noanomalies['Datetime'].dt.date]).mean().reset_index()

df_aqi_daily.rename(columns={'Datetime': 'Date'}, inplace=True)
print('Number of records:', len(df_aqi_daily))
display(pd.concat([df_aqi_daily.head(2), df_aqi_daily.tail(2)]))


# %%
# Grouping observations by month and monitoring point to end up with one record for each month and monitoring station
df_aqi_monthly_station = df_aqi_noanomalies.groupby(
    [df_aqi_noanomalies['Datetime'].dt.month, 'Station']).mean(numeric_only=False)
df_aqi_monthly_station.drop(['Datetime'], axis=1, inplace=True)
df_aqi_monthly_station = df_aqi_monthly_station.reset_index()

calendar = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

df_aqi_monthly_station.rename(columns={'Datetime': 'Month'}, inplace=True)
df_aqi_monthly_station['Month'] = df_aqi_monthly_station['Month'].map(calendar)

print('Number of records:', len(df_aqi_monthly_station))
display(pd.concat([df_aqi_monthly_station.head(
    2),  df_aqi_monthly_station.tail(2)]))


# %%
# Grouping observations by individual hours of the day and monitoring point to end up with one record for each particular hour
df_aqi_hour_station = df_aqi_noanomalies.groupby(
    [df_aqi_noanomalies['Datetime'].dt.hour, 'Station']).mean(numeric_only=False)
df_aqi_hour_station.drop(['Datetime'], axis=1, inplace=True)
df_aqi_hour_station = df_aqi_hour_station.reset_index()

df_aqi_hour_station.rename(columns={'Datetime': 'Hour'}, inplace=True)
df_aqi_hour_station['Hour'] = df_aqi_hour_station['Hour'] + 1

print('Number of records:', len(df_aqi_hour_station))
display(pd.concat([df_aqi_hour_station.head(2),  df_aqi_hour_station.tail(2)]))


# %%
# Grouping observations by individual days of the week and monitoring point to end up with one record for each day of the week
df_aqi_dayweek_station = df_aqi_noanomalies.groupby(
    [df_aqi_noanomalies['Datetime'].dt.dayofweek, 'Station']).mean(numeric_only=False)
df_aqi_dayweek_station.drop(['Datetime'], axis=1, inplace=True)
df_aqi_dayweek_station = df_aqi_dayweek_station.reset_index()

df_aqi_dayweek_station.rename(
    columns={'Datetime': 'Day of week'}, inplace=True)
df_aqi_dayweek_station['Day of week'] = df_aqi_dayweek_station['Day of week'] + 1

print('Number of records:', len(df_aqi_dayweek_station))
display(pd.concat([df_aqi_dayweek_station.head(
    2),  df_aqi_dayweek_station.tail(2)]))


# %%
# Grouping observations by monitoring point to end up with one record for each station
df_aqi_station = df_aqi_noanomalies.groupby(
    [df_aqi_noanomalies['Station']]).mean().reset_index()
#df_aqi_station.drop(['Datetime'], axis = 1, inplace = True)

print('Number of records:', len(df_aqi_station))
display(pd.concat([df_aqi_station.head(2),  df_aqi_station.tail(2)]))


# %% [markdown]
# # **<font color="#ffb94f">2.0 VISUALIZATION</font>**

# %% [markdown]
# Having a first look at visualizing data with different granularities, first with all the data available, then separated by monthly means of the measurements and then daily. Alternative groupings such as by hour of the day or day of the week are analyzed to see if there's any trend therein

# %%
# Looking at temp measurement from a certain station with all the data available
fig, ax = plt.subplots(figsize=(24, 4))
ax = sns.lineplot(data=df_aqi_noanomalies[df_aqi_noanomalies['Station']
                  == 'Atemajac'], x='Datetime', y='Temperature', lw=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
plt.title('Temperature values across time (1 hour window between measurements)')
plt.ylim(5, 36)
plt.show()

# Looking at temp measurement from a certain station with the data grouped by daily measurements
fig, ax = plt.subplots(figsize=(24, 4))
ax = sns.lineplot(
    data=df_aqi_daily_station[df_aqi_daily_station['Station'] == 'Atemajac'], x='Date', y='Temperature', lw=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
plt.title('Temperature values across the year (daily)')
plt.ylim(5, 36)
plt.show()

# Looking at temp measurement from a certain station with the data grouped by monthly measurements
fig, ax = plt.subplots(figsize=(24, 4))
ax = sns.lineplot(
    data=df_aqi_monthly_station[df_aqi_monthly_station['Station'] == 'Atemajac'], x='Month', y='Temperature', lw=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.title('Temperature values across the year (monthly)')
plt.ylim(5, 36)
plt.show()


# %% [markdown]
# As seen from the previous plot, different granularities capture different aspects of the data, considering that there are 8 stations to analyze, using all data might prove to be to dense for visualization and time series analysis, the monthly grouping on the other hand is very clear and simple, but perhaps too much, it might be useful to visualize general trends. Finally, the daily grouping hits just the sweet spot for visualizing and analyzing temporal data
#

# %%
# Looking at the trends in the different monthly measurements across all year with on every station
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(24, 12))
k = 0
for i in range(4):
    for j in range(2):
        sns.lineplot(ax=ax[i, j], data=df_aqi_monthly_station, x='Month',
                     y=list_columns[k], hue='Station', palette='Paired', lw=1)
        ax[i, j].set(title=list_columns[k])
        ax[i, j].legend([], [], frameon=False)
        ax[i, j].xaxis.set_major_locator(ticker.MultipleLocator(1))
        fig.suptitle('Values depending on the month', fontsize=24)
        fig.tight_layout(pad=2.0)
        k += 1

plt.legend(bbox_to_anchor=(1, 1))
plt.show()


# %%
# Plotting aggregated values across each different hour of the day for each station
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(24, 12))
k = 0
for i in range(4):
    for j in range(2):
        sns.lineplot(ax=ax[i, j], data=df_aqi_hour_station, x='Hour',
                     y=list_columns[k], hue='Station', palette='Paired', lw=1)
        ax[i, j].set(title=list_columns[k])
        ax[i, j].legend([], [], frameon=False)
        ax[i, j].xaxis.set_major_locator(ticker.MultipleLocator(4))
        fig.suptitle('Values depending on the hour of the day', fontsize=24)
        fig.tight_layout(pad=2.0)
        k += 1

plt.legend(bbox_to_anchor=(1, 1))
plt.show()


# %%
# Plotting aggregated values across each different day of the week for each station
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(24, 12))
k = 0
for i in range(4):
    for j in range(2):
        sns.lineplot(ax=ax[i, j], data=df_aqi_dayweek_station, x='Day of week',
                     y=list_columns[k], hue='Station', palette='Paired', lw=1)
        ax[i, j].set(title=list_columns[k])
        ax[i, j].legend([], [], frameon=False)
        ax[i, j].xaxis.set_major_locator(ticker.MultipleLocator(1))
        fig.suptitle('Values depending on the day of the week', fontsize=24)
        fig.tight_layout(pad=2.0)
        k += 1

plt.legend(bbox_to_anchor=(1, 1))
plt.show()


# %%
# Using plotly to plot interactive lineplots for daily measurements of all values, with plotly, it is possible to select which stations are shown on the plot
for feature in list_columns:
    fig = px.line(x=df_aqi_daily['Date'],
                  y=df_aqi_daily[feature], title=feature)
    fig.update_layout(title_text=feature +
                      ' daily aggregated values', title_x=0.5)
    fig.update_layout(width=1400, height=300, margin=dict(
        t=50, b=10, l=10, r=10), xaxis_title='', yaxis_title=feature)
    fig.show()


# %%
# Using plotly to plot interactive lineplots for daily measurements of all values, with plotly, it is possible to select which stations are shown on the plot
palette = list(sns.color_palette(palette='Paired',
               n_colors=len(list_monitor_station)).as_hex())

for feature in list_columns:
    fig = go.Figure()
    for station, p in zip(list_monitor_station, palette):
        fig.add_trace(go.Scatter(x=df_aqi_daily_station[df_aqi_daily_station['Station'] == station]['Date'],
                                 y=df_aqi_daily_station[df_aqi_daily_station['Station']
                                                        == station][feature],
                                 name=station,
                                 line_color=p))
    fig.update_layout(title_text=feature +
                      ' daily values by station', title_x=0.5)
    fig.update_layout(autosize=False, width=1400, height=400,
                      margin=dict(t=50, b=10, l=10, r=10))
    fig.show()


# %%
# Plotting facet grids to compare each station values to the rest of the stations
for feature in list_columns:
    grid = sns.relplot(data=df_aqi_daily_station, x="Date", y=feature,
                       col="Station", hue="Station",
                       kind="line", palette="Spectral",
                       linewidth=3, zorder=4, col_wrap=4,
                       height=3, aspect=1.5, legend=False)

    # add text and silhouettes
    for time, ax in grid.axes_dict.items():
        ax.text(.1, .85, time, transform=ax.transAxes, fontweight="bold")
        sns.lineplot(data=df_aqi_daily_station, x="Date", y=feature, units="Station",
                     estimator=None, color=".7", linewidth=1, ax=ax)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    grid.set_titles("")
    grid.fig.suptitle(
        '{} values across time separated by monitoring station'.format(feature))
    grid.set_axis_labels("", "{}".format(feature))
    grid.tight_layout(pad=2.0)


# %%
# Radial plots
for feature in list_columns:
    fig = go.Figure()

    for station, p in zip(list_monitor_station, palette):
        fig.add_trace(go.Scatterpolar(r=df_aqi_monthly_station[df_aqi_monthly_station['Station'] == station][feature],
                                      theta=df_aqi_monthly_station[df_aqi_monthly_station['Station']
                                                                   == station]['Month'],
                                      name=station, marker=dict(color=p)))
    fig.update_layout(title_text=feature +
                      ' monthly values by station', title_x=0.5)
    fig.update_layout(showlegend=True, width=600, height=500,
                      margin=dict(t=50, b=50, l=50, r=50))
    fig.show()


# %% [markdown]
# # **<font color="#ffb94f">3.0 GEOGRAPHIC ANALYSIS</font>**

# %%
# Official administrative divisions to use as tessellations for each geographical area
url_area = 'https://raw.githubusercontent.com/Bruno-Limon/air-quality-analysis/main/area-ZMG.geojson'

area_base = gpd.read_file(url_area)
area_merged = gpd.GeoSeries(unary_union(area_base['geometry']))

# Loading official city boundaries
official_tessellation = gpd.read_file(url_area)
official_tessellation.insert(
    0, 'tile_ID', range(0, len(official_tessellation)))
official_tessellation['mun_name'] = official_tessellation['mun_name'].str[0]
official_tessellation = official_tessellation.set_crs(
    {'init': 'epsg:4326'}, allow_override=True)


# %%
# Observing official municipal tessellation
tess_color = "tab20b"
official_tessellation.plot(cmap=tess_color)
plt.title('Official tessellation')
plt.axis('off')
plt.show()

official_tessellation.head()


# %%
# Points to build voronoi tessellation, locations of monitoring stations
dict_station_coord = {'Oblatos': (20.700501, -103.296648),
                      'Loma Dorada': (20.631665, -103.256809),
                      'Atemajac': (20.719626, -103.355412),
                      'Miravalle': (20.614511, -103.343352),
                      'Las Pintas': (20.576708, -103.326533),
                      'Las Aguilas': (20.630983, -103.416735),
                      'Tlaquepaque': (20.640941, -103.312497),
                      'Vallarta': (20.680141, -103.398572),
                      }

# Creating polygon of city boundaries for each area using shapely's function unary_union
area_boundary = unary_union(area_base.geometry)
voronoi_points = np.array([[lng, lat]
                          for (lat, lng) in dict_station_coord.values()])
region_polys, region_pts = voronoi_regions_from_coords(
    voronoi_points, area_boundary)
voronoi_tessellation = gpd.GeoDataFrame(
    columns=['geometry'], crs={'init': 'epsg:4326'})
voronoi_tessellation['name'] = [
    'cell ' + str(i) for i in range(1, len(region_polys) + 1)]
voronoi_tessellation['geometry'] = [region_polys[index]
                                    for index, row in voronoi_tessellation.iterrows()]
voronoi_tessellation['lat'] = [
    lat for (lat, lng) in dict_station_coord.values()]
voronoi_tessellation['lng'] = [
    lng for (lat, lng) in dict_station_coord.values()]

voronoi_tessellation['Station'] = [caseta for caseta,
                                   (lat, lng) in dict_station_coord.items()]
voronoi_tessellation = voronoi_tessellation.explode().reset_index(
    drop=True).drop([1, 2]).reset_index(drop=True)


# %%
gdf_voronoi_points = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(voronoi_points[:, 0], voronoi_points[:, 1]))

# Observing voronoi tessellation produced by using the locations of monitoring stations
tess_color = "tab20b"
fig, ax = plt.subplots(nrows=1, ncols=1)
voronoi_tessellation.plot(ax=ax, cmap=tess_color)
gdf_voronoi_points.plot(ax=ax, color='white', markersize=10)
ax.axis('off')
plt.title('Voronoi tessellation')
plt.show()

voronoi_tessellation


# %%
# Merging voronoi tessellation with the data grouped by station so we can have
gdf_merged_station = voronoi_tessellation.set_index('Station').join(
    df_aqi_station.set_index('Station')).reset_index()
print('Number of records:', len(gdf_merged_station))
display(pd.concat([gdf_merged_station.head(2),  gdf_merged_station.tail(2)]))


# %%
# Plotting geodata for each cell of the voronoi tessellation, with the color corresponding to the
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 9))
k = 0
for i in range(2):
    for j in range(4):
        gdf_merged_station.plot(
            ax=ax[i, j], column=list_columns[k], cmap='RdBu_r', linewidth=1, edgecolor='gray')
        ax[i, j].set(title=list_columns[k])
        ax[i, j].axis('off')
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=min(
            gdf_merged_station[list_columns[k]]), vmax=max(gdf_merged_station[list_columns[k]])))
        fig.colorbar(sm, ax=ax[i, j])
        fig.suptitle(
            'Mean values for each station aggregated throughout the year', fontsize=24)
        fig.tight_layout()
        k += 1


# %%
# Plotting the previous plot this time with a slider to select between different months to filter the data shown
voronoi_tessellation_json = voronoi_tessellation.copy()
voronoi_tessellation_json.index = voronoi_tessellation['Station']
# voronoi_tessellation_json.drop(['name', 'lat', 'Station', 'lng'], axis=1, inplace=True)
voronoi_tessellation_json = json.loads(voronoi_tessellation_json.to_json())

# Choropleth map
for value in list_columns:
    fig = px.choropleth_mapbox(data_frame=df_aqi_monthly_station,
                               geojson=voronoi_tessellation_json,
                               locations=df_aqi_monthly_station.Station,
                               color=value,
                               center={'lat': 20.621236, 'lon': -103.355412},
                               mapbox_style='carto-positron',
                               zoom=8,
                               color_continuous_scale='RdBu_r',
                               range_color=(min(df_aqi_monthly_station[value]), max(
                                   df_aqi_monthly_station[value])),
                               animation_frame='Month',
                               title=value,
                               opacity=.5,
                               width=1400,
                               height=600)
    fig.show()


# %%
# geolocator = Nominatim(timeout = 10, user_agent = 'BrunoLimon')
# location = geolocator.geocode('Buckingham Palace')
# location


# %%
ox_factories = ox.geometries_from_place(
    "Guadalajara, Jalisco", {'landuse': 'industrial'}).to_crs('epsg:4326')
ox_factories2 = ox.geometries_from_place(
    "Guadalajara, Jalisco", {'building': 'industrial'}).to_crs('epsg:4326')
ox_schools = ox.geometries_from_place(
    "Guadalajara, Jalisco", {'building': 'school'}).to_crs('epsg:4326')

factory_locations = [[ox_factories.geometry.centroid.x[i],
                      ox_factories.geometry.centroid.y[i]] for i in range(len(ox_factories))]
factory_locations2 = [[ox_factories2.geometry.centroid.x[i],
                       ox_factories2.geometry.centroid.y[i]] for i in range(len(ox_factories2))]
school_locations = [[ox_schools.geometry.centroid.x[i],
                     ox_schools.geometry.centroid.y[i]] for i in range(len(ox_schools))]
factory_locations += factory_locations2


# %%
print('Number of factories found: ', len(factory_locations))
print('Number of schools found: ', len(school_locations))


# %%
map_f = folium.Map(location=[20.66682, -103.39182],
                   zoom_start=10, tiles='CartoDB positron')

for _, row in gdf_merged_station.iterrows():
    sim_geo = gpd.GeoSeries(row['geometry']).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {
                           'fillColor': 'None', 'color': 'Blue'})
    folium.Circle([row['lat'], row['lng']], radius=20, weight=10).add_to(geo_j)
    geo_j.add_to(map_f)

mc_factory = MarkerCluster()
for i in range(len(factory_locations)):
    mc_factory.add_child(folium.Marker(
        [factory_locations[i][1], factory_locations[i][0]])).add_to(map_f)
    folium.Circle([factory_locations[i][1], factory_locations[i][0]],
                  radius=20,
                  color='red').add_to(map_f)

mc_school = MarkerCluster()
for i in range(len(school_locations)):
    mc_school.add_child(folium.Marker(
        [school_locations[i][1], school_locations[i][0]])).add_to(map_f)
    folium.Circle([school_locations[i][1], school_locations[i][0]],
                  radius=20,
                  color='green').add_to(map_f)
map_f


# %%
map_f = folium.Map(location=[20.66682, -103.39182],
                   zoom_start=10, tiles='CartoDB positron')

for _, row in gdf_merged_station.iterrows():
    sim_geo = gpd.GeoSeries(row['geometry']).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {
                           'fillColor': 'None', 'color': 'Blue'})
    folium.Circle([row['lat'], row['lng']], radius=20, weight=10).add_to(geo_j)
    geo_j.add_to(map_f)

for i in range(len(factory_locations)):
    folium.Circle([factory_locations[i][1], factory_locations[i][0]],
                  radius=20,
                  color='red').add_to(map_f)

for i in range(len(school_locations)):
    folium.Circle([school_locations[i][1], school_locations[i][0]],
                  radius=20,
                  color='green').add_to(map_f)

HeatMap(list(zip(np.array(factory_locations)[:, 1], np.array(
    factory_locations)[:, 0])), radius=10).add_to(map_f)
map_f


# %% [markdown]
# # **<font color="#ffb94f">4.0 DIMENSIONALITY REDUCTION</font>**

# %% [markdown]
# ## **<font color="#84f745">3.1 PCA</font>**

# %%
pca = PCA(n_components=8)
Principal_components = pca.fit_transform(X_train)

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


# %%
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

levels, categories = pd.factorize(df_aqi_noanomalies['Station'])
colors = [plt.cm.Set3(i) for i in levels]
handles = [matplotlib.patches.Patch(color=plt.cm.Set3(
    i), label=c) for i, c in enumerate(categories)]

fig = plt.figure(figsize=(12, 4))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
            c=colors, edgecolor='k', alpha=.6, linewidth=.7)
plt.legend(handles=handles)
plt.show()


# %%
pca = PCA(n_components=3)
aqi_pca = pca.fit_transform(X_train)

x = aqi_pca[:, 0]
y = aqi_pca[:, 1]
z = aqi_pca[:, 2]

levels, categories = pd.factorize(df_aqi_noanomalies['Station'])
colors = [plt.cm.Set3(i) for i in levels]
handles = [matplotlib.patches.Patch(color=plt.cm.Set3(
    i), label=c) for i, c in enumerate(categories)]

fig = plt.figure(figsize=(15, 10))
fig.tight_layout()
ax = plt.axes(projection="3d")
ax.scatter3D(x, y, z, c=colors)
plt.legend(handles=handles)
plt.show()


# %% [markdown]
# ## **<font color="#84f745">3.2 FEATURE SELECTION</font>**

# %%
print('Raw variances')
df_aqi_noanomalies.iloc[:, 1:9].var()


# %%
print('Normalized variances')
(df_aqi_noanomalies.iloc[:, 1:9]/df_aqi_noanomalies.iloc[:, 1:9].mean()).var()


# %%
sel_var = VarianceThreshold(threshold=.1).fit(
    df_aqi_noanomalies.iloc[:, 1:9]/df_aqi_noanomalies.iloc[:, 1:9].mean())

selected = sel_var.get_support()
print('Full list of features: ', list_columns)
print('Features with at least .1 of normalized variance: ',
      np.array(list_columns)[selected])


# %%
for i, feature in enumerate(list_columns):
    sel_mod = SelectFromModel(LinearRegression()).fit(
        X_train[:, [x for x in range(X_train.shape[1]) if x not in [i]]], X_train[:, i])
    selected = sel_mod.get_support()
    print('Relevant features to predict {} '.format(feature), np.array(
        [x for x in list_columns if x != feature])[selected])


# %% [markdown]
# # **<font color="#ffb94f">5.0 REGRESSION</font>**

# %%


# %% [markdown]
# # **<font color="#ffb94f">6.0 TIME SERIES ANALYSIS</font>**

# %%
df_aqi_daily_norm = (df_aqi_daily.iloc[:, 1:9] - df_aqi_daily.iloc[:, 1:9].min()) / (
    df_aqi_daily.iloc[:, 1:9].max() - df_aqi_daily.iloc[:, 1:9].min())

dict_distances_ts = {}
for i in range(0, 8):
    for j in range(0, 8):
        ts1 = df_aqi_daily_norm.iloc[:, i]
        ts2 = df_aqi_daily_norm.iloc[:, j]
        dict_distances_ts[(list_columns[i], list_columns[j])
                          ] = round(euclidean(ts1, ts2), 4)

dict_distances_ts = {key: val for key,
                     val in dict_distances_ts.items() if val != 0}
dict_distances_wo_duplicates = {}
for key, val in dict_distances_ts.items():
    if val not in dict_distances_wo_duplicates.values():
        dict_distances_wo_duplicates[key] = val

k_smallest = dict(
    sorted(dict_distances_wo_duplicates.items(), key=lambda x: x[1])[:5])
k_smallest


# %%
feature1 = 'CO'
feature2 = 'NO2'
index1 = list_columns.index(feature1)
index2 = list_columns.index(feature2)

ts1 = df_aqi_daily_norm.iloc[:, index1]
ts2 = df_aqi_daily_norm.iloc[:, index2]

print('Distance between {} and {}:'.format(
    list_columns[index1], list_columns[index2]), round(euclidean(ts1, ts2), 4))

fig, ax = plt.subplots(figsize=(24, 4))
plt.plot(ts1, label=list_columns[index1])
plt.plot(ts2, label=list_columns[index2])
plt.title('Original values (normalized) across time (monthly)')
plt.xticks([])
plt.legend(loc='best')
plt.ylim(0, 1)
plt.show()


# %%
ts1 = df_aqi_daily_norm.iloc[:, index1]
ts2 = df_aqi_daily_norm.iloc[:, index2]
ts1 = ts1 - ts1.mean()
ts2 = ts2 - ts2.mean()
ts1 = (ts1 - ts1.min()) / (ts1.max() - ts1.min())
ts2 = (ts2 - ts2.min()) / (ts2.max() - ts2.min())

print('Distance between {} and {}:'.format(
    list_columns[index1], list_columns[index2]), round(euclidean(ts1, ts2), 4))

fig, ax = plt.subplots(figsize=(24, 4))
plt.plot(ts1, label=list_columns[index1])
plt.plot(ts2, label=list_columns[index2])
plt.title('Offset Translation')
plt.xticks([])
plt.legend(loc='best')
plt.show()


# %%
ts1 = df_aqi_daily_norm.iloc[:, index1]
ts2 = df_aqi_daily_norm.iloc[:, index2]
ts1 = (ts1 - ts1.mean()) / ts1.std()
ts2 = (ts2 - ts2.mean()) / ts2.std()
ts1 = (ts1 - ts1.min()) / (ts1.max() - ts1.min())
ts2 = (ts2 - ts2.min()) / (ts2.max() - ts2.min())

print('Distance between {} and {}:'.format(
    list_columns[index1], list_columns[index2]), round(euclidean(ts1, ts2), 4))

fig, ax = plt.subplots(figsize=(24, 4))
plt.plot(ts1, label=list_columns[index1])
plt.plot(ts2, label=list_columns[index2])
plt.title('Amplitude Scaling')
plt.xticks([])
plt.legend(loc='best')
plt.show()


# %%
w = 3
ts1 = df_aqi_daily_norm.iloc[:, index1]
ts2 = df_aqi_daily_norm.iloc[:, index2]

fig, ax = plt.subplots(figsize=(24, 4))
plt.plot(ts1.rolling(window=w).mean(), label="Time Series 1 trend")
plt.plot(ts2.rolling(window=w).mean(), label="Time Series 2 trend")
plt.title('Trends')
plt.xticks([])
plt.legend(loc='best')
plt.show()


# %%
w = 5
ts1 = df_aqi_daily_norm.iloc[:, index1]
ts2 = df_aqi_daily_norm.iloc[:, index2]
ts1 = ts1 - ts1.rolling(window=w).mean()
ts2 = ts2 - ts2.rolling(window=w).mean()
ts1 = (ts1 - ts1.min()) / (ts1.max() - ts1.min())
ts2 = (ts2 - ts2.min()) / (ts2.max() - ts2.min())

ts1[np.isnan(ts1)] = 0
ts2[np.isnan(ts2)] = 0

print('Distance between {} and {}:'.format(
    list_columns[index1], list_columns[index2]), round(euclidean(ts1, ts2), 4))

fig, ax = plt.subplots(figsize=(24, 4))
plt.plot(ts1, label=list_columns[index1])
plt.plot(ts2, label=list_columns[index2])
plt.title('Removing trends')
plt.xticks([])
plt.legend(loc='best')
plt.show()


# %%
ts1 = df_aqi_daily_norm.iloc[:, index1]
ts2 = df_aqi_daily_norm.iloc[:, index2]

# Offset translation
ts1 = ts1 - ts1.mean()
ts2 = ts2 - ts2.mean()

# Amplitude Scaling
ts1 = (ts1 - ts1.mean()) / ts1.std()
ts2 = (ts2 - ts2.mean()) / ts2.std()

w = 5
# Removing trends
ts1 = ts1 - (ts1.rolling(window=w).mean())
ts2 = ts2 - (ts2.rolling(window=w).mean())

# Removing noise
ts1 = ((ts1 - ts1.mean()) / ts1.std()).rolling(window=w).mean()
ts2 = ((ts2 - ts2.mean()) / ts2.std()).rolling(window=w).mean()

# Final normalization
ts1 = (ts1 - ts1.min()) / (ts1.max() - ts1.min())
ts2 = (ts2 - ts2.min()) / (ts2.max() - ts2.min())

ts1[np.isnan(ts1)] = 0
ts2[np.isnan(ts2)] = 0

print('Distance between {} and {}:'.format(
    list_columns[index1], list_columns[index2]), round(euclidean(ts1, ts2), 4))

fig, ax = plt.subplots(figsize=(24, 4))
plt.plot(ts1, label=list_columns[index1])
plt.plot(ts2, label=list_columns[index2])
plt.title('Combining transformation methods')
plt.xticks([])
plt.legend(loc='best')
plt.show()



