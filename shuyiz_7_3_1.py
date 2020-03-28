'''
assignment1: Change of Trend Detection
question1:take years 1 and 2. For each month, compute the ”candi- date” days and decide whether there is a significant change of pricing trend in each month. Use 0.1 as critical value.
'''
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import f as fisher_f

def linear_model(x,y):
    weights = np.polyfit(x,y, 1)
    model = np.poly1d(weights)
    predicted = model(x)
    mse = mean_squared_error(y,predicted)
    return mse

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
        mse = linear_model(np.array(range(1,len(data)+1)),data.loc[index_start:index_end,'Adj Close'].values)

        minimal = []

        for k in range(3,len(data)):
            data1_x = np.array(range(1,k))
            data1_y = data.loc[index_start:index_start+k-2,'Adj Close'].values
            mse1 = linear_model(data1_x,data1_y)
            data2_x = np.array(range(k,len(data)+1) )
            data2_y = data.loc[index_start+k-1:index_end,'Adj Close'].values
            mse2 = linear_model(data2_x,data2_y)
            if minimal == [] or minimal[0] > mse1+mse2:
                minimal = [mse1+mse2,mse,len(data)]

        F = ((minimal[1]-minimal[0])/2)/(minimal[0]/(minimal[2]-4))
        p_value = fisher_f.cdf(F, 2, minimal[2]-4)
        if 1-p_value < 0.1:
            print('Year',year,' month',month,'has significant change of price trend')
        else:
            print('Year',year,' month',month,'has not significant change of price trend')

 
