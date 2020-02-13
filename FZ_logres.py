
# coding: utf-8

# ## 로지스틱 회귀 적용

# In[2]:


import pandas as pd
import numpy as np
import os, sys, time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


# ### 데이터세트 로딩

# In[3]:


FILENAME = 'FZ_Data/FZ_all_ffill.csv'

if not os.path.exists(FILENAME):
    assert False, 'cannot find FILENAME: {}'.format(FILENAME)
else:
    print('found : {}'.format(FILENAME))


# In[4]:


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


# In[5]:


df.columns


# In[6]:


use_colmns = ['지점', '지점명', '일시', '기온(°C)', '강수량(mm)', 
       '풍속(m/s)', '풍향(16방위)', '습도(%)', 
       '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)',
       '일조(hr)', '일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)',
       '전운량(10분위)', '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )', '시정(10m)',
       '지면상태(지면상태코드)', '현상번호(국내식)', '지면온도(°C)',
       'FZ_flag']

df = df[use_colmns]


# In[7]:


df.shape


# ## 한글 컬럼이름들을 영문으로 변경함

# In[16]:


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


# In[17]:


df.columns


# In[18]:


df.dtypes


# In[19]:


df.groupby(['FZ_flag']).size()


# ### Pandas correlation matrix

# In[34]:


df.corr()


# In[35]:


def heatmap(data, x, y, size):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    n_colors = 256 # Use 256 colors for the diverging color palette
    palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1)) # target index in the color palette
        return palette[ind]

    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size * size_scale,
        c=data.apply(value_to_color), # Vector of square color values, mapped to color palette
        marker='s'
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)


# In[38]:


columns = df.columns
corr = df[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    data=corr['value'],
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# ## Statsmodel 를 사용한 로지스틱 회귀

# In[20]:


def build_formula(x_list, y):
    x_str = ' + '.join(x_list)
    return '{} ~ {}'.format(y, x_str)


# In[21]:


x_features = ['Temp', 'Rain', 'WindSpeed', 'Moist', 'hPa', 'DewTemp', 'CurhPa', 
              'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10', 'SurfaceTemp']
y_column = 'FZ_flag'


# In[27]:


x_features = ['Temp', 'Rain', 'WindSpeed', 'Moist', 'hPa', 'DewTemp', 'CurhPa', 
              'SnowCm', 'Snow3hr', 'MClouds_10', 'SurfaceTemp']
y_column = 'FZ_flag'


# In[28]:


fm_str = build_formula(x_features, y_column)


# In[29]:


fm_str


# In[30]:


model_all = sm.Logit.from_formula(fm_str, df)


# In[31]:


result = model_all.fit(maxiter=100)


# In[32]:


print(result.summary())


# In[33]:


print(result.summary2())


# ### 변수 scaling
# 
# * MinMaxScaler

# In[39]:


feature_cols = ['Date', 'Temp', 'Rain', 'WindSpeed', 'WindDir',
       'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 'Daylight', 'DaylightMJ',
       'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10', 'CloudDesigns_Abb',
       'HClouds_100m', 'Visibility_10m', 'GroundState_Code', 'PhenomenaNo',
       'SurfaceTemp']
feature_cols_numeric_only = ['Temp', 'Rain', 'WindSpeed', 'WindDir',
       'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 'Daylight', 'DaylightMJ',
       'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10',
       'HClouds_100m', 'Visibility_10m', 
       'SurfaceTemp']


# In[40]:


X = df.loc[:, feature_cols_numeric_only]
y = df['FZ_flag'].values


# In[41]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(X)
X_sc = scaler.fit_transform(X)


# In[42]:


df_X_sc = pd.DataFrame(data=X_sc, columns=feature_cols_numeric_only)
df_y = pd.DataFrame(data=y, columns=['FZ_flag'])


# In[43]:


df_sc = pd.concat([df_X_sc, df_y], axis=1)


# In[44]:


df_sc.corr()


# In[46]:


columns = df_sc.columns
corr = df_sc[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    data=corr['value'],
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# In[51]:


x_features = ['Temp', 'Rain', 'WindSpeed', 'Moist', 'hPa', 'DewTemp', 'CurhPa', 
              'SnowCm', 'MClouds_10', 'SurfaceTemp']
y_column = 'FZ_flag'


# In[52]:


fm_str = build_formula(x_features, y_column)


# In[53]:


fm_str


# In[54]:


model_sc = sm.Logit.from_formula(fm_str, df_sc)


# In[55]:


result = model_sc.fit(maxiter=100)


# In[56]:


print(result.summary2())


# ## sci-kit learn Logistic Regression (all columns)

# In[25]:


df_seoul = df.loc[df['Office_Name']=='서울']


# In[50]:


feature_cols = ['Date', 'Temp', 'Rain', 'WindSpeed', 'WindDir',
       'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 'Daylight', 'DaylightMJ',
       'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10', 'CloudDesigns_Abb',
       'HClouds_100m', 'Visibility_10m', 'GroundState_Code', 'PhenomenaNo',
       'SurfaceTemp']
feature_cols_numeric_only = ['Temp', 'Rain', 'WindSpeed', 'WindDir',
       'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 'Daylight', 'DaylightMJ',
       'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10',
       'HClouds_100m', 'Visibility_10m', 
       'SurfaceTemp']


# In[51]:


X = df_seoul.loc[:, feature_cols_numeric_only]
y = df_seoul['FZ_flag'].values


# In[95]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(X)
X_sc = scaler.fit_transform(X)


# In[107]:


df_X_sc = pd.DataFrame(data=X_sc, columns=['Temp', 'Rain', 'WindSpeed', 'WindDir',
       'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 'Daylight', 'DaylightMJ',
       'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10',
       'HClouds_100m', 'Visibility_10m', 
       'SurfaceTemp'])
df_y = pd.DataFrame(data=y, columns=['FZ_flag'])


# In[109]:


df_sc = pd.concat([df_X_sc, df_y], axis=1)


# In[110]:


df_sc.shape


# In[96]:


df_seoul.corr()


# In[97]:


def heatmap(data, x, y, size):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    n_colors = 256 # Use 256 colors for the diverging color palette
    palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1)) # target index in the color palette
        return palette[ind]

    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size * size_scale,
        c=data.apply(value_to_color), # Vector of square color values, mapped to color palette
        marker='s'
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)


# In[98]:


columns = feature_cols + ['FZ_flag'] 
corr = df_seoul[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    data=corr['value'],
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# In[99]:


logreg = LogisticRegression().fit(X, y)


# In[100]:


logreg.coef_


# ## statsmodel

# In[101]:


import statsmodels.api as sm


# In[102]:


df_seoul.isnull().sum()


# In[103]:


df_seoul.dtypes


# In[104]:


def build_formula(x_list, y):
    x_str = ' + '.join(x_list)
    return '{} ~ {}'.format(y, x_str)


# In[105]:


fm_str = build_formula(['Temp', 'Rain', 'WindSpeed', 
                        'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 
                        'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10',
                        'HClouds_100m', 'Visibility_10m', 
                        'SurfaceTemp'], 'FZ_flag')


# ## HClouds_100m, SeahPa --> singular matrix

# In[106]:


fm_str = build_formula(['Temp', 'Rain', 'WindSpeed', 
                        'Moist', 'hPa', 'DewTemp', 'CurhPa', 
                        'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10',
                        'SurfaceTemp'], 'FZ_flag')


# In[89]:


model = sm.Logit.from_formula(fm_str, df_seoul)


# In[90]:


result = model.fit(maxiter=100)
print(result.summary())


# In[94]:


print(result.summary2())


# In[93]:


print(np.exp(result.params)) # 오즈 비(Odds Ratio) 출력


# ## after scaling

# In[115]:


df_sc.head()


# In[113]:


model_sc = sm.Logit.from_formula(fm_str, df_sc)


# In[114]:


result2 = model_sc.fit(maxiter=100)
print(result2.summary2())

