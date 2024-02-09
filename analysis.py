import pandas as pd
import math

data = pd.read_csv('아파트(매매)__실거래가_20230916101223.csv', sep=',', encoding='cp949', skiprows=15)

tmp = data['시군구'].value_counts().reset_index()

data['시도'] = data['시군구'].apply(lambda x : x.split(' ')[0])


tmp = data[['시도', '계약일']].value_counts().reset_index()

tmp = data.pivot_table(index='시도', columns='계약일', values='거래금액(만원)', aggfunc='count')

data.dtypes

data['거래금액(만원)'] = data['거래금액(만원)'].str.replace(',', '').astype('int')

data.dtypes

tmp = data.pivot_table(index='시도', columns='계약일', values='거래금액(만원)', aggfunc='mean')

tmp2 = data[data['건축년도'].isnull()]
data = data[~data['건축년도'].isnull()]

data['건축년도_구분'] = data['건축년도'].apply(lambda x : math.floor(x /10) * 10)

tmp = data.pivot_table(columns='시도', index='건축년도_구분', values='거래금액(만원)', aggfunc='mean')


data.dtypes

data['단가'] = data['거래금액(만원)'] / data['전용면적(㎡)']

tmp = data.pivot_table(columns='시도', index='건축년도_구분', values='단가', aggfunc='mean')


