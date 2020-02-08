# -*- coding: utf-8 -*-
'''
assigment 3: test normality returns
question 1: compute the mean and standard deviation of daily returns
'''
import os
import pandas as pd

ticker = 'ZSAN'
input_dir = r'/Users/zhengshuyi'
ticker_file = os.path.join(input_dir, ticker + '.csv')
plot_dir = r'/Users/zhengshuyi'

try:
    df = pd.read_csv(ticker_file)
    data = []
    for i in range(2015,2020):
        df_year = df[df['Year'] == i]
        days = df_year.shape[0]
        mean = df_year['Return'].mean()
        sd = df_year['Return'].std()
        less = df_year[df_year['Return'] < mean-2*sd].shape[0]
        more = df_year[df_year['Return'] > mean+2*sd].shape[0]
        actual_days = less+more
        predict_days = round(0.05*days,0)
        data.append([i, actual_days, predict_days])
    title = ['year','actual_outliers','predict_outliers']
    result = pd.DataFrame(data,columns = title)
    print(result)

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)