
# coding: utf-8

# ## LSTM Logistic Regresion
# 
# * n_hr = 6

# In[1]:


from math import sqrt
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense, LSTM

import datetime as dt
import pandas as pd
import numpy as np
import os


# In[2]:


debug = True
desired_width = 250 
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


# In[3]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """ 
    Frame a time series as a supervised learning dataset.
    Arguments:
    :data: Sequence of observations as a list or NumPy array.
    :n_in: Number of lag observations as input (X).
    :n_out: Number of observations as output (y).
    :dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    :Pandas DataFrame of series framed for supervised learning.
    """
    # convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # debug
    if debug:
        print('n_vars = {}, df.shape = {}'.format(n_vars, df.shape))

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg 


# In[4]:


def show_dataframe(df, head=5):
    if head > 0:
        print('------------- Head({}) --------------'.format(head))
        print(df.head(head))

    print('------------- Columns -----------------')
    for col in df.columns.tolist():
        if df[col].dtype in ['float32', 'int32', 'float64', 'int64']:
            print('  {:15s} = {:10.3f} ~ {:10.3f}, NaN = {}'.format(col, df[col].min(), df[col].max(), df[col].isnull().sum()))


# ## data loading

# In[5]:


FILENAME = 'FZ_Data/FZ_all_ffill.csv'

if not os.path.exists(FILENAME):
    assert False, 'cannot find FILENAME: {}'.format(FILENAME)
else:
    print('found : {}'.format(FILENAME))


# In[6]:


if FILENAME.endswith('.xlsx'):
    df = pd.read_excel(FILENAME, 
                       header=0,
                       encoding='euc-kr',
                       converters={'지점':str, '지면상태(지면상태코드)':str, '현상번호(국내식)':str})
elif FILENAME.endswith('.csv'):
    df = pd.read_csv(FILENAME, 
                     header=0,
                     encoding='euc-kr',
                     converters={'지점':str, '지면상태(지면상태코드)':str, '현상번호(국내식)':str})


# In[7]:


use_colmns = ['지점', '지점명', '일시', '기온(°C)', '강수량(mm)', 
       '풍속(m/s)', '풍향(16방위)', '습도(%)', 
       '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)',
       '일조(hr)', '일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)',
       '전운량(10분위)', '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )', '시정(10m)',
       '지면상태(지면상태코드)', '현상번호(국내식)', '지면온도(°C)',
       'FZ_flag']

df = df[use_colmns]


# In[8]:


df.rename(columns = {'지점' : 'Office',
                      '지점명' : 'Office_Name',
                      '일시' : 'Date',
                      '기온(°C)': 'Temp', 
                      '강수량(mm)': 'Rain', 
                      '풍속(m/s)': 'WindSpeed', 
                      '풍향(16방위)': 'WindDir',  
                      '습도(%)': 'Moist',
                      '증기압(hPa)': 'hPa', 
                      '이슬점온도(°C)': 'DewTemp', 
                      '현지기압(hPa)': 'CurhPa', 
                      '해면기압(hPa)': 'SeahPa',
                      '일조(hr)': 'Daylight', 
                      '일사(MJ/m2)': 'DaylightMJ', 
                      '적설(cm)': 'SnowCm', 
                      '3시간신적설(cm)': 'Snow3hr',
                      '전운량(10분위)': 'Clouds_10', 
                      '중하층운량(10분위)': 'MClouds_10',                      
                      '운형(운형약어)' : 'CloudDesigns_Abb',                     
                      '최저운고(100m )': 'HClouds_100m', 
                      '시정(10m)': 'Visibility_10m', 
                      '지면상태(지면상태코드)': 'GroundState_Code',
                      '현상번호(국내식)': 'PhenomenaNo',
                      '지면온도(°C)': 'SurfaceTemp'
                     }, inplace=True)


# In[9]:


df.head()


# ## re-ordering columns

# In[11]:


reorderd_list = ['Office', 'Date', 'FZ_flag',
                 'Temp', 'Rain', 'WindSpeed', 'WindDir', 'Moist', 'hPa', 'DewTemp', 'CurhPa',
                 'SeahPa', 'Daylight', 'DaylightMJ', 'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10',
                 'HClouds_100m', 'SurfaceTemp']


# In[12]:


df = df[reorderd_list]


# In[13]:


n_features = len(df.columns.tolist()) - 1
n_ignore = len(df.columns.tolist()) - 3


# In[14]:


def reframed_for_lstm(df, n_steps=6, n_features=10, n_ignore=5, include_weather=False):
    # get list of Office
    stations = df['Office'].unique()
    if debug:
        print('reframed for LSTM')
        print('n_features = {}'.format(n_features))
        print('target {} stations: {}'.format(len(stations), stations))

    data = [ ] 
    i = 0 
    for station in stations:
        if debug and i == 0:
            print('#{}: station = {}'.format(i, station))

        df1 = df[df['stationid'] == station]
        df1 = df1.drop(columns=df1.columns[0], axis=1)

        if debug and i == 0:
            print('Shape = {}, Columns = {}'.format(df1.shape, df1.columns.tolist()))

        # integer encode weather (non integer/float type)
        if include_weather:
            encoder = LabelEncoder()
            df1.loc[:, 'weather'] = encoder.fit_transform(df1.loc[:, 'weather'])

        # ensure all data is float
        values = df1.values
        values = values.astype('float64')

        if debug and i == 0:
            dd = pd.DataFrame(values)
            print('before scaled = shape {}'.format(dd.shape))
            #print(dd.head(100))
            #print(dd.tail(200))
            #print('station: {} -> {}'.format(station, values[1, :]))

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(values)
        scaled = scaler.transform(values)

        # frame as supervised learning
        if debug and i == 0:
            dd = pd.DataFrame(scaled)
            print('scaled columns = #{} -> {}'.format(len(dd.columns.tolist()), dd.columns.tolist()))

        reframed = series_to_supervised(scaled, n_steps, 1)
        if debug and i == 0:
            print("after reframed = shape {}".format(reframed.shape))
            #print(reframed.tail(5))

        # drop columns we don't want to predict (last utctime ~ weather)
        reframed.drop(reframed.columns[[range(-n_ignore, 0, 1)]], axis=1, inplace=True)
        if debug and i == 0:
            print('reframed columns = #{} -> {}'.format(len(reframed.columns.tolist()), reframed.columns.tolist()))
            print("after  re-framed.drop = shape {}".format(reframed.shape))
            #print(reframed.tail(5))

        item = {}
        item['station'] = station
        item['data'] = reframed
        item['scaler'] = scaler
        data.append(item)

        i += 1
    else:
        if debug:
            print('load {} stations data'.format(len(stations)))
        return data

