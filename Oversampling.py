
# coding: utf-8

# ## SMOTE oversampling
# 
# * imbalance

# In[55]:


import pandas as pd
import numpy as np
import os, sys, time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import accuracy_score  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv('FZ_Data/FZ_all_ffill.csv', 
                 header=0,
                 encoding='euc-kr',
                 converters={'지점':str, '지면상태(지면상태코드)':str, '현상번호(국내식)':str})


# In[3]:


use_columns = ['지점', '지점명', '일시', '기온(°C)', '강수량(mm)', 
       '풍속(m/s)', '풍향(16방위)', '습도(%)', 
       '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)',
       '일조(hr)', '일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)',
       '전운량(10분위)', '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )', '시정(10m)',
       '지면상태(지면상태코드)', '현상번호(국내식)', '지면온도(°C)',
       'FZ_flag']

df = df[use_columns]


# df.rename(columns = {'지점' : 'Office',
#                       '지점명' : 'Office_Name',
#                       '일시' : 'Date',
#                       '기온(°C)': 'Temp', 
#                       '강수량(mm)': 'Rain', 
#                       '풍속(m/s)': 'WindSpeed', 
#                       '풍향(16방위)': 'WindDir',  
#                       '습도(%)': 'Moist',
#                       '증기압(hPa)': 'hPa', 
#                       '이슬점온도(°C)': 'DewTemp', 
#                       '현지기압(hPa)': 'CurhPa', 
#                       '해면기압(hPa)': 'SeahPa',
#                       '일조(hr)': 'Daylight', 
#                       '일사(MJ/m2)': 'DaylightMJ', 
#                       '적설(cm)': 'SnowCm', 
#                       '3시간신적설(cm)': 'Snow3hr',
#                       '전운량(10분위)': 'Clouds_10', 
#                       '중하층운량(10분위)': 'MClouds_10',                      
#                       '운형(운형약어)' : 'CloudDesigns_Abb',                     
#                       '최저운고(100m )': 'HClouds_100m', 
#                       '시정(10m)': 'Visibility_10m', 
#                       '지면상태(지면상태코드)': 'GroundState_Code',
#                       '현상번호(국내식)': 'PhenomenaNo',
#                       '지면온도(°C)': 'SurfaceTemp'
#                      }, inplace=True)

# In[4]:


df.groupby('FZ_flag').size()


# In[11]:


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


# In[12]:


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


# In[13]:


def get_Xy(df, X_features, y_col):
    if not isinstance(X_features, list):
        X_features = [X_features]
    if not isinstance(y_col, list):
        y_col = [y_col]
    names = X_features + y_col
    for name in names:
        if name not in df.columns:
            assert False, 'cannot find such column: {}'.format(name)
    X = df.loc[:, X_features]
    y = df[y_col].values
    print('** X : {} rows, {} columns'.format(X.shape[0], X.shape[1]))
    print('** y : {} rows'.format(len(y)))
    return X, y


# ### over sampling

# In[97]:


def oversampling(df, X_features, y_col, SPLIT_FIRST=True, OVERSAMPLE=True, OVERSAMPLE_RATIO=1.0):
    from sklearn.preprocessing import MinMaxScaler
    from imblearn.over_sampling import SMOTE

    X, y = get_Xy(df, X_features, y_col)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(X)
    X = scaler.fit_transform(X)
    
    if not OVERSAMPLE:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        return X_tr, X_te, y_tr, y_te

    # 모델설정
    smote = SMOTE(random_state=1234, sampling_strategy=OVER_SAMPLE_RATIO)

    # train데이터를 넣어 복제함
    if SPLIT_FIRST:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        
        X_tr, y_tr = smote.fit_sample(X_tr, list(y_tr))
        X_te, y_te = smote.fit_sample(X_te, list(y_te))

        print('After OverSampling, the shape of train_X: {}'.format(X_tr.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(y_tr.shape))

        print("After OverSampling, counts of train label '1': {}".format(sum(y_tr==1)))
        print("After OverSampling, counts of train label '0': {}".format(sum(y_tr==0)))

        print('After OverSampling, the shape of test_X: {}'.format(X_te.shape))
        print('After OverSampling, the shape of test_y: {} \n'.format(y_te.shape))

        print("After OverSampling, counts of test label '1': {}".format(sum(y_te==1)))
        print("After OverSampling, counts of test label '0': {}".format(sum(y_te==0)))
    else:
        X, y = smote.fit_sample(X,list(y))

        print('After OverSampling, the shape of X: {}'.format(X.shape))
        print('After OverSampling, the shape of y: {} \n'.format(y.shape))

        print("After OverSampling, counts of y label '1': {}".format(sum(y==1)))
        print("After OverSampling, counts of y label '0': {}".format(sum(y==0)))
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        
        print('After OverSampling, the shape of train_X: {}'.format(X_tr.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(y_tr.shape))

        print("After OverSampling, counts of train label '1': {}".format(sum(y_tr==1)))
        print("After OverSampling, counts of train label '0': {}".format(sum(y_tr==0)))

        print('After OverSampling, the shape of test_X: {}'.format(X_te.shape))
        print('After OverSampling, the shape of test_y: {} \n'.format(y_te.shape))

        print("After OverSampling, counts of test label '1': {}".format(sum(y_te==1)))
        print("After OverSampling, counts of test label '0': {}".format(sum(y_te==0)))
        
    df_tr = pd.concat([pd.DataFrame(data=X_tr, columns=X_features), 
                       pd.DataFrame(data=y_tr, columns=[y_col])], axis=1)

    df_te = pd.concat([pd.DataFrame(data=X_te, columns=X_features), 
                       pd.DataFrame(data=y_te, columns=[y_col])], axis=1)

    return X_tr, X_te, y_tr, y_te, df_tr, df_te


# In[82]:


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


# In[83]:


def build_formula(x_list, y):
    x_str = ' + '.join(x_list)
    return '{} ~ {}'.format(y, x_str)


# ### all Office

# In[84]:


X_tr, X_te, y_tr, y_te, df_tr, df_te = oversampling(df, feature_cols_numeric_only, 'FZ_flag')


# In[85]:


df_tr.corr()


# In[86]:


df_tr_sorted = df_tr.corr()['FZ_flag'][:-1].sort_values(ascending=False)
df_tr_sorted


# In[87]:


columns = feature_cols_numeric_only + ['FZ_flag']
corr = df_tr[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    data=corr['value'],
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# In[88]:


x_features = ['Temp', 'Rain', 'WindSpeed', 'WindDir',
       'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 'Daylight', 'DaylightMJ',
       'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10', 
       'HClouds_100m', 'Visibility_10m', 
       'SurfaceTemp']
y_column = 'FZ_flag'


# In[89]:


fm_str = build_formula(x_features, y_column)
fm_str


# In[90]:


df_tr.shape


# In[91]:


model_all = sm.Logit.from_formula(fm_str, df_tr)


# In[92]:


result = model_all.fit(maxiter=100, method='bfgs')


# In[93]:


result.summary2()


# ### 학습 정확도

# In[106]:


t = result.pred_table()
print(t)
print("Accuracy:",np.diag(t).sum()/t.sum())


# In[94]:


X_te.shape


# ### 테스트 정확도

# In[108]:


y_pred_proba = result.predict(df_te)
y_pred = [ 0 if x < 0.5 else 1 for x in y_pred_proba]


# In[109]:


score = accuracy_score(y_te, y_pred)
score


# ### 대관령 Office

# In[117]:


df_daekwa = df.loc[df['Office_Name']=='대관령']


# In[118]:


X_tr, X_te, y_tr, y_te, df_tr, df_te = oversampling(df_daekwa, feature_cols_numeric_only, 'FZ_flag')


# In[119]:


df_tr.corr()


# In[120]:


df_tr_sorted = df_tr.corr()['FZ_flag'][:-1].sort_values(ascending=False)
df_tr_sorted


# In[121]:


columns = feature_cols_numeric_only + ['FZ_flag']
corr = df_tr[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    data=corr['value'],
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# In[122]:


x_features = ['Temp', 'Rain', 'WindSpeed', 'WindDir',
       'Moist', 'hPa', 'DewTemp', 'CurhPa', 'SeahPa', 'Daylight', 'DaylightMJ',
       'SnowCm', 'Snow3hr', 'Clouds_10', 'MClouds_10', 
       'HClouds_100m', 'Visibility_10m', 
       'SurfaceTemp']
y_column = 'FZ_flag'


# In[123]:


fm_str = build_formula(x_features, y_column)
fm_str


# In[91]:


fm_str


# In[124]:


df_tr.shape


# In[125]:


model_2 = sm.Logit.from_formula(fm_str, df_tr)


# In[126]:


result2 = model_2.fit(maxiter=100, method='bfgs')


# In[128]:


result2.summary2()


# In[129]:


t = result.pred_table()
print(t)
print("Accuracy:",np.diag(t).sum()/t.sum())


# In[130]:


y_pred_proba = result.predict(df_te)
y_pred = [ 0 if x < 0.5 else 1 for x in y_pred_proba]


# In[131]:


score = accuracy_score(y_te, y_pred)
score

