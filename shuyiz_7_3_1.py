'''

'''
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import f as finsher_f

def linear_model(x,y):
    weights = np.polyfit(x,y, 1)
    model = np.poly1d(weights)
    predicted = model(x)
    rmse = np.sqrt(mean_squared_error(y,predicted))
    return rmse

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)

for year in range(2017,2019):
    for month in range(1,12):

        index_start = df[(df['Year'] == year) & (df['Month'] == month)].index.values.astype(int)[0]
        index_end = df[(df['Year'] == year) & (df['Month'] == month+1)].index.values.astype(int)[0]-1

        data = df[(df['Year'] == year) & (df['Month'] == month)]
        dic = {}

        for n in range(3,len(data)):
            data1_x = np.array(range(1,n))
            data1_y = data.loc[index_start:index_start+n-2,'Adj Close'].values
            rmse1 = linear_model(data1_x,data1_y)
            data2_x = np.array(range(n,len(data)+1))
            data2_y = data.loc[index_start+n-1:index_end,'Adj Close'].values
            rmse2 = linear_model(data2_x,data2_y)
            dic[n] = rmse1+rmse2
            minimal = min(dic.values)
        print(dic)

