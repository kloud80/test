import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

data = pd.read_csv('아파트(매매)__실거래가_20230916101223.csv', sep=',', encoding='cp949', skiprows=15)
data['거래금액(만원)'] = data['거래금액(만원)'].str.replace(',', '').astype('int')
data = data[~data['건축년도'].isnull()]
data['단가'] = data['거래금액(만원)'] / data['전용면적(㎡)']
data['년한'] = 2024 - data['건축년도']
data['시도'] = data['시군구'].apply(lambda x : x.split(' ')[0])
data['년한제곱'] = data['년한']  * data['년한']


data = data[data['시도'].isin(['서울특별시', '경기도', '인천광역시'])]

data.columns

lr = LinearRegression()

X = np.array(data[['년한', '년한제곱',  '전용면적(㎡)', '층']].values)
X_dummy = np.array(pd.get_dummies(data['시도']).values)
X = np.concatenate([X, X_dummy],axis=1)
Y = np.array(data['단가'].values)



if X.ndim == 1 : X = X[:,np.newaxis]
if Y.ndim == 1 : Y = Y[:,np.newaxis]

lr.fit(X, Y)
lr.coef_
lr.intercept_
Y2 = lr.predict(X)

plt.scatter(X[:,0], Y)
plt.scatter(X[:,0], Y2, color='red')
plt.title('y = {}*x + {}'.format(lr.coef_[0], lr.intercept_))
plt.show()


X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
