
# coding: utf-8

# ## Decision Tree 이용한 Classification (Logistic Regressoin)

# In[1]:


import pandas as pd
import numpy as np
import os, sys, time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree


# ### 데이터세트

# In[68]:


FILENAME = 'FZ_Data/FZ_all_ffill.csv'

if not os.path.exists(FILENAME):
    assert False, 'cannot find FILENAME: {}'.format(FILENAME)
else:
    print('found : {}'.format(FILENAME))


# In[84]:


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


# In[85]:


use_colmns = ['지점', '지점명', '일시', '기온(°C)', '강수량(mm)', 
       '풍속(m/s)', '풍향(16방위)', '습도(%)', 
       '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)',
       '일조(hr)', '일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)',
       '전운량(10분위)', '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )', '시정(10m)',
       '지면상태(지면상태코드)', '현상번호(국내식)', '지면온도(°C)',
       'FZ_flag']

df = df[use_colmns]


# In[86]:


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


# In[87]:


df.groupby(['FZ_flag']).size()


# ### Feature Columns

# In[88]:


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


# In[89]:


def get_Xy(df, x_features):
    X = df.loc[:, x_features]
    y = df['FZ_flag'].values
    return X, y


# In[90]:


X, y = get_Xy(df, feature_cols_numeric_only)


# In[91]:


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)


# In[92]:


sum(y_tr==1), sum(y_tr==0)


# In[93]:


sum(y_te==1), sum(y_te==0)


# ### 오버샘플링

# In[94]:


OVER_SAMPLE = True
OVER_SAMPLE_RATIO = 1.0


# In[95]:


if OVER_SAMPLE:
    from sklearn.preprocessing import MinMaxScaler
    from imblearn.over_sampling import SMOTE

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(X)
    X = scaler.fit_transform(X)

    # 모델설정
    smote = SMOTE(random_state=1234, sampling_strategy=OVER_SAMPLE_RATIO)

    # train데이터를 넣어 복제함
    X_tr, y_tr = smote.fit_sample(X_tr,list(y_tr))
    X_te, y_te = smote.fit_sample(X_te,list(y_te))

    print('After OverSampling, the shape of train_X: {}'.format(X_tr.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_tr.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_tr==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_tr==0)))
    
    print('After OverSampling, the shape of train_X: {}'.format(X_te.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_te.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_te==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_te==0)))


# ### Decesion Tree

# In[96]:


from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[98]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_tr, y_tr)


# ### 정확도는 

# In[99]:


y_predict = clf.predict(X_te)
accuracy_score(y_te, y_predict)


# In[ ]:


#if OVER_SAMPLE:
#    X_tr, X_te, y_tr, y_te = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)
#else:
#    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)


# In[100]:


clf.decision_path


# ### 규칙

# In[101]:


tree.plot_tree(clf) 


# In[102]:


import graphviz


# In[104]:


dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=feature_cols_numeric_only,  
                                class_names=['0', '1'],  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# In[106]:


graph.render('DecisionTree.png')


# In[ ]:


graph.save()

