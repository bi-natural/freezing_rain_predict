
# coding: utf-8

# ## 기상청 종관지상관측 데이터 (ASOS)
# 
# * https://data.kma.go.kr

# In[2]:


import pandas as pd
import os, sys, time
import glob


# In[46]:


#FILENAME = 'FZ_Data/ASOS_All.xlsx'
FILENAME = 'FZ_Data/ASOS_All_csv.csv'

if not os.path.exists(FILENAME):
    assert False, 'cannot find FILENAME: {}'.format(FILENAME)
else:
    print('found : {}'.format(FILENAME))


# ### load ASOS 데이터

# In[83]:


if FILENAME.endswith('.xlsx'):
    df = pd.read_excel(FILENAME, 
                       encoding='euc-kr',
                       converters={'지점':str, '지면상태(지면상태코드)':str, '현상번호(국내식)':str})
elif FILENAME.endswith('.csv'):
    df = pd.read_csv(FILENAME, 
                       encoding='euc-kr',
                       converters={'지점':str, '지면상태(지면상태코드)':str, '현상번호(국내식)':str})


# In[84]:


df.shape


# In[85]:


df.columns


# ### 24개 컬럼이 분석 대상
# 
# * 데이터품질에 대한 내용인 QC플래그 제외
# * 특정지점에서만 데이터를 취합하는 지중온도는 제외 

# In[86]:


use_columns = ['지점', '지점명', '일시', '기온(°C)', '강수량(mm)', 
       '풍속(m/s)', '풍향(16방위)', '습도(%)', 
       '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)',
       '일조(hr)', '일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)',
       '전운량(10분위)', '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )', '시정(10m)',
       '지면상태(지면상태코드)', '현상번호(국내식)', '지면온도(°C)']


# In[87]:


len(use_columns)


# In[88]:


df = df[use_columns]


# ### 결측치 확인

# In[89]:


df.isnull().sum()


# In[90]:


df.groupby('현상번호(국내식)').size()


# In[91]:


df.head()


# ### 결측치 처리: 숫자가 0인 것들이 데이터가 없는 경우들
# 
# * 강수량(mm)이 확인 안 된 경우 
# * 일조(hr)이 확인 안 된 경우
# * 적설(cm)이 확인 안 된 경우
# * 3시간신적설(cm)이 확인 안 된 경우

# In[92]:


for col in ['강수량(mm)', '일조(hr)', '일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)']:
    df[col] = df[col].fillna(0)


# In[93]:


df[['강수량(mm)', '일조(hr)', '일사(MJ/m2)','적설(cm)', '3시간신적설(cm)']].isnull().sum()


# In[94]:


df_copy = df.copy()


# ### 결측치 처리: 주변값을 활용하여 비슷한 것으로 처리
# 
# * 습도(%)
# * 3시간마다 측정하는 값들: 증기압(hPa), 이슬점온도(°C), 현지기압(hPa), 해면기압(hPa), 전운량(10분위), 중하층운량(10분위), 운형(운형약어),
#   최저운고(100m ), 시정(10m), 지면상태(지면상태코드), 지면온도(°C)

# In[95]:


IMPUTE_METHOD=['ffill', 'bfill']


# In[96]:


def impute_columns(df, cols, impute_methods):
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(impute_methods, str):
        impute_methods = [impute_methods]
    for col in cols:
        if col not in df.columns:
            print('cannot find such column: {}'.format(col))
            continue
            
        for method in impute_methods:
            before = df[col].isnull().sum()
            if before > 0:
                if method == 'ffill':
                    df[col] = df[col].fillna(method='ffill')
                    after = df[col].isnull().sum()
                    print('** impute: {} column {} -> {}'.format(col, before, after))
                elif method == 'bfill':
                    df[col] = df[col].fillna(method='bfill')
                    after = df[col].isnull().sum()
                    print('** impute: {} column {} -> {}'.format(col, before, after))
                elif method == 'linear_interpolate':
                    df[col] = df[col].interpolate(method='linear')
                    after = df[col].isnull().sum()
                    print('** impute: {} column {} -> {}'.format(col, before, after))
                else:
                    print('cannot find such impute method')     


# In[97]:


impute_columns(df, ['습도(%)'], IMPUTE_METHOD)


# In[98]:


df['습도(%)'].isnull().sum()


# In[99]:


impute_columns(df, ['증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '전운량(10분위)', 
                    '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', 
                    '지면온도(°C)'], IMPUTE_METHOD)


# In[100]:


df[['증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '전운량(10분위)', 
   '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', '지면온도(°C)']].isnull().sum()


# In[101]:


df.isnull().sum()


# ### 현상번호(국내식)을 FZ_flag로 변환
# 
# * 3, 7, 12 : 어는 비(Freezing Rain)
# * 15, 18 : 얼음침 

# In[104]:


def is_FZ_rain(x):
    if pd.isna(x):
        return 0
    x = str(x)
    if len(x) <= 2:
        if x in ['3', '7', '12', '15', '18']:
            return 1
    else:
        xx = [x[i:i+2] for i in range(0, len(x), 2)]
        for x in xx:
            if x in ['03', '07', '12', '15', '18']:
                return 1
    return 0


# In[105]:


df['FZ_flag'] = df['현상번호(국내식)'].map(is_FZ_rain)


# In[106]:


df.groupby('FZ_flag').size()


# ### ffill 방식으로 처리한 것을 저장
# 
# * FZ_all_ffill.csv

# In[107]:


df.to_csv('FZ_Data/FZ_all_ffill.csv', index=False, encoding='euc-kr')


# ### interpolate 사용

# In[108]:


df_copy.isnull().sum()


# In[109]:


IMPUTE_METHOD='linear_interpolate'


# In[110]:


impute_columns(df_copy, ['습도(%)'], IMPUTE_METHOD)


# In[111]:


impute_columns(df_copy, ['증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '전운량(10분위)', 
                    '중하층운량(10분위)', '운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', 
                    '지면온도(°C)'], IMPUTE_METHOD)


# In[115]:


df_copy.isnull().sum()


# ### numeric type이 아닌 것은 interpolate가 동작하지 않음

# In[116]:


impute_columns(df_copy, ['운형(운형약어)'], ['ffill', 'bfill'])


# In[120]:


df_copy.isnull().sum()


# In[121]:


impute_columns(df_copy, ['지면온도(°C)'], ['ffill', 'bfill'])


# In[122]:


df_copy.isnull().sum()


# In[117]:


df_copy['FZ_flag'] = df_copy['현상번호(국내식)'].map(is_FZ_rain)


# In[118]:


df_copy.groupby('FZ_flag').size()


# In[123]:


df_copy.to_csv('FZ_Data/FZ_all_interpolate.csv', index=False, encoding='euc-kr')


# ### Test codes

# In[244]:


df['증기압(hPa)'].head(10)


# In[245]:


#df['증기압(hPa)'] = df['증기압(hPa)'].interpolate(method='linear')
df['증기압(hPa)'] = df['증기압(hPa)'].fillna(method='ffill')


# In[246]:


df['증기압(hPa)'].head(10)


# In[247]:


df['시정(10m)'].head(10)


# In[248]:


df['시정(10m)'] = df['시정(10m)'].interpolate(method='linear')


# In[249]:


df['시정(10m)'].head(10)


# In[251]:


df['일사(MJ/m2)'].head(20)


# In[252]:


df['일사(MJ/m2)'] = df['일사(MJ/m2)'].fillna(0)


# In[253]:


df['일사(MJ/m2)'].head(20)


# In[254]:


df['3시간신적설(cm)'].head(20)


# In[256]:


dfdf.groupby(['운형(운형약어)']).size()


# In[258]:


df['운형(운형약어)'].head(20)


# In[259]:


df['운형(운형약어)'] = df['운형(운형약어)'].fillna(method='ffill')


# In[266]:


df['운형(운형약어)'].isnull().sum()


# In[260]:


df['운형(운형약어)'].head(20)


# In[261]:


df['지면상태(지면상태코드)'].head(20)


# In[262]:


df['지면상태(지면상태코드)'] = df['지면상태(지면상태코드)'].fillna(method='ffill')


# In[265]:


df['지면상태(지면상태코드)'].isnull().sum()


# In[264]:


df['지면상태(지면상태코드)'].head(20)


# In[267]:


df['지면상태(지면상태코드)'] = df['지면상태(지면상태코드)'].fillna(method='bfill')


# In[268]:


df['지면상태(지면상태코드)'].isnull().sum()


# In[269]:


df['이슬점온도(°C)'].head(20)


# In[ ]:


IMPUTE_METHOD=['ffill', 'bfill']


# In[276]:


def impute_columns(dataframes, cols, impute_methods):
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(impute_methods, str):
        impute_methods = [impute_methods]
    for col in cols:
        if col not in dataframes.columns:
            print('cannot find such column: {}'.format(col))
            continue
            
        for method in impute_methods:
            before = dataframes[col].isnull().sum()
            if before > 0:
                if method == 'ffill':
                    dataframes[col] = dataframes[col].fillna(method='ffill')
                    after = dataframes[col].isnull().sum()
                    print('** impute: {} column {} -> {}'.format(col, before, after))
                elif method == 'bfill':
                    dataframes[col] = dataframes[col].fillna(method='bfill')
                    after = dataframes[col].isnull().sum()
                    print('** impute: {} column {} -> {}'.format(col, before, after))
                elif method == 'linear_interpolate':
                    dataframes[col] = dataframes[col].interpolate(method='linear')
                    after = dataframes[col].isnull().sum()
                    print('** impute: {} column {} -> {}'.format(col, before, after))
                else:
                    print('cannot find such impute method')                


# In[277]:


impute_columns(df, ['이슬점온도(°C)'], ['ffill', 'bfill'])


# In[278]:


df['이슬점온도(°C)'].isnull().sum()


# In[280]:


impute_columns(df, '해면기압(hPa)', 'linear_interpolate')


# In[168]:


for col in ['이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)']:
    df[col] = df[col].fillna(method='ffill')


# In[169]:


for col in ['지면온도(°C)', '전운량(10분위)', '중하층운량(10분위)', '최저운고(100m )']:
    #df[col] = df[col].fillna(method='ffill')
    df[col] = df[col].interpolate(method='linear')


# In[170]:


df.isnull().sum()


# In[171]:


for col in ['일사(MJ/m2)', '3시간신적설(cm)', '지면상태(지면상태코드)', '지면온도(°C)']:
    df[col] = df[col].fillna(method='bfill')


# In[172]:


df.groupby(['현상번호(국내식)']).size()


# In[173]:


def is_fzrain(x):
    if pd.isna(x):
        return 0
    x = str(x)
    if len(x) <= 2:
        if x in ['3', '7', '12', '15', '18']:
            return 1
    else:
        xx = [x[i:i+2] for i in range(0, len(x), 2)]
        for x in xx:
            if x in ['03', '07', '12', '15', '18']:
                return 1
    return 0


# In[174]:


is_fzrain('1')


# In[175]:


is_fzrain('3'), is_fzrain('03')


# In[176]:


is_fzrain('12'), is_fzrain('7112401901')


# In[177]:


df['FZ_flag'] = df['현상번호(국내식)'].map(is_fzrain)


# In[178]:


df.groupby(['FZ_flag']).size()


# In[180]:


df.to_csv('FZ_Data/FZ_all2.csv', index=False, encoding='euc-kr')


# In[181]:


df.describe()


# In[182]:


df1 = df.loc[df['지점명']=='대관령']


# In[183]:


df1.corr()


# In[184]:


df2 = df.loc[df['지점명']=='영월']


# In[185]:


df2.corr()


# In[186]:


df3 = df.loc[df['지점명']=='서울']


# In[233]:


df3.corr()


# In[188]:


from sklearn.linear_model import LogisticRegression


# In[200]:


feature_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)',  '습도(%)', 
       '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)',
       '일조(hr)', '일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)',
       '전운량(10분위)', '중하층운량(10분위)', '최저운고(100m )', '시정(10m)', '지면온도(°C)']


# In[213]:


df3.isnull().sum()


# In[201]:


X = df3.loc[:, feature_cols]
y = df3['FZ_flag'].values


# In[202]:


logreg = LogisticRegression().fit(X, y)


# In[203]:


df3.rename(columns = {'기온(°C)': 'Temp', 
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
                      '전운량(10분위)': 'A', 
                      '중하층운량(10분위)': 'B', 
                      '최저운고(100m )': 'C', 
                      '시정(10m)': 'D', 
                      '지면온도(°C)': 'SurfaceTemp'
                     }, inplace=True)


# In[212]:


df3.columns


# In[206]:


import statsmodels.api as sm


# In[210]:


#model = sm.Logit.from_formula('FZ_flag ~ Temp + Rain + WindSpeed + WindDir + Moist + hPa + DewTemp + CurhPa + SeahPa + Daylight + DaylightMJ + SnowCm + Snow3hr + A + B + C + D + SurfaceTemp', df3)


# In[228]:


df3.groupby('SurfaceTemp').size()


# In[229]:


df3.groupby('FZ_flag').size()


# In[231]:


df.dtypes


# In[220]:


model = sm.Logit.from_formula('FZ_flag ~ Temp + Rain + WindSpeed + Moist + hPa + CurhPa + SurfaceTemp', df3)


# In[230]:


result_med = model.fit(maxiter=100)
print(result_med.summary())

