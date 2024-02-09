import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#%%
from keras.models import Sequential #개별 레이어른 선형적으로 적제하기 위한 모델
from keras.layers import Dense #일반적인 형태의 뉴럴네트워크 계층 / 앞선 학습에 사용한 은닉/출력층에 해당
from keras import optimizers

from sklearn.metrics import r2_score



data = pd.read_csv('아파트(매매)__실거래가_20230916101223.csv', sep=',', encoding='cp949', skiprows=15)
data['거래금액(만원)'] = data['거래금액(만원)'].str.replace(',', '').astype('int')
data = data[~data['건축년도'].isnull()]
data['단가'] = data['거래금액(만원)'] / data['전용면적(㎡)']
data['년한'] = 2024 - data['건축년도']
data['시도'] = data['시군구'].apply(lambda x : x.split(' ')[0])
data['년한제곱'] = data['년한']  * data['년한']


# data = data[data['시도'].isin(['서울특별시', '경기도', '인천광역시'])]

data.columns


X = np.array(data[['년한', '년한제곱',  '전용면적(㎡)', '층']].values)
X_dummy = np.array(pd.get_dummies(data['시도']).values)
X = np.concatenate([X, X_dummy],axis=1)
Y = np.array(data['단가'].values)



if X.ndim == 1 : X = X[:,np.newaxis]
if Y.ndim == 1 : Y = Y[:,np.newaxis]


model = Sequential() #모델을 선언한다
model.add(Dense(units=100, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.summary() #모델 요약


history  = model.fit(X, Y, epochs=50)

Y2 = model.predict(X)


print(r2_score(Y, Y2))


plt.scatter(X[:,0], Y)
plt.scatter(X[:,0], Y2, color='red')
plt.show()