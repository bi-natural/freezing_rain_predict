
# coding: utf-8

# ## 이전 N -t 시간을 고려한 Logistic Regression

# In[3]:


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


# In[281]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ### shift 를 활용한 데이터 만들기

# In[15]:


debug = True


# In[204]:


def series_to_timeseries(data, x_features, n_step=6):
    # convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # debug
    if debug:
        print('n_vars = {}, df.shape = {}'.format(n_vars, df.shape))

    # input sequence (t-n, ... t-1)
    for i in range(n_step, -1, -1):
        cols.append(df.shift(i))
        names += [('{}_t{}'.format(j, i)) for j in x_features]

    # forecast sequence (t, t+1, ... t+n)
    #for i in range(0, n_out):
    #    cols.append(df.shift(-i))
    #    if i == 0:
    #        names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    #    else:
    #        names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    return agg 


# ### data loading

# In[244]:


FILENAME = 'FZ_Data/FZ_all_ffill.csv'

if not os.path.exists(FILENAME):
    assert False, 'cannot find FILENAME: {}'.format(FILENAME)
else:
    print('found : {}'.format(FILENAME))


# In[245]:


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


# In[246]:


use_colmns = ['지점', '지점명', '일시', '기온(°C)', '강수량(mm)', 
       '풍속(m/s)', '풍향(16방위)', '습도(%)', 
       '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)',
       '일조(hr)', '일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)',
       '전운량(10분위)', '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )', '시정(10m)',
       '지면상태(지면상태코드)', '현상번호(국내식)', '지면온도(°C)',
       'FZ_flag']

df = df[use_colmns]


# In[247]:


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


# In[248]:


df.columns


# In[249]:


x_features = ['Temp', 'Rain', 'WindSpeed', 'WindDir',
       'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 'Daylight', 'DaylightMJ',
       'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10', 'CloudDesigns_Abb',
       'HClouds_100m', 'Visibility_10m', 'GroundState_Code', 'PhenomenaNo',
       'SurfaceTemp']


# In[250]:


df_x_features = df[x_features]


# In[251]:


N_STEP=3


# In[252]:


reframed = series_to_timeseries(df_x_features, x_features, n_step=N_STEP)


# In[253]:


reframed.columns


# In[254]:


reframed.head()


# In[255]:


reframed.shape


# In[256]:


df_head = df[['Office', 'Office_Name', 'Date']]
df_tail = df[['FZ_flag']]


# In[257]:


df_reframed = pd.concat([df_head, reframed, df_tail], axis=1)


# In[258]:


df_reframed.shape


# In[259]:


df_reframed.head()


# In[260]:


df_reframed.isnull().sum()


# In[261]:


df_reframed.dropna(inplace=True)


# In[262]:


df_reframed.isnull().sum()


# In[263]:


df_reframed.shape


# In[264]:


df_reframed.to_csv('FZ_Data/FZ_all_reframed.csv', index=False, encoding='euc-kr')


# ## correlation

# In[265]:


df_reframed.corr()


# In[266]:


df_sorted = df_reframed.corr()['FZ_flag'][:-1].sort_values(ascending=False)


# In[282]:


df_sorted


# In[279]:


df_sorted.plot(figsize=(20,10))


# ### use Logistic Regression with reframed

# In[269]:


from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


# In[270]:


def build_formula(x_list, n_step, y):
    names = list()
    for i in range(n_step, -1, -1):
        names += [('{}_t{}'.format(j, i)) for j in x_features]
        
    x_str = ' + '.join(names)
    return '{} ~ {}'.format(y, x_str)


# In[283]:


x_features = ['Temp', 'Rain', 'WindSpeed', 'WindDir',
       'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 'Daylight', 'DaylightMJ',
       'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10', 
       'HClouds_100m', 'Visibility_10m',  'SurfaceTemp']
y_column = 'FZ_flag'


# In[284]:


fm_str = build_formula(x_features, n_step=N_STEP, y=y_column)


# In[285]:


fm_str


# In[286]:


model_all = sm.Logit.from_formula(fm_str, df_reframed)


# In[300]:


result = model_all.fit(maxiter=800, method='bfgs')


# In[301]:


result.summary2()

