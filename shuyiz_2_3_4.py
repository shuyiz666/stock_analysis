# -*- coding: utf-8 -*-
'''
assigment 3: test normality returns
question 1: Summarize findings in a table for each year and discuss  findings
'''
import os
import pandas as pd

ticker = 'ZSAN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
plot_dir = wd

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
        data.append([i, days, round(mean,4), round(sd,2),str(round(less/days,2)*100)+'%',str(round(more/days,2)*100)+'%'])
    title = ['year','trading days','u','sd','% days < u-2*sd','% days > u+2*sd']
    result = pd.DataFrame(data,columns = title)
    print(result)
    print('')
    print('Year 2018 is closest to normal distribution since the sum of % days < u-2*sd and days > u+2*sd is close to 5%.')
    print('In each year, % days > u+2*sd is more than or equal to % days < u-2*sd.')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)