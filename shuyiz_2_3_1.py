'''
assigment 3: test normality returns
question 1: compute the number of days with positive and negative returns
'''

import os
import pandas as pd

ticker = 'ZSAN'
input_dir = r'/Users/zhengshuyi'
ticker_file = os.path.join(input_dir, ticker + '.csv')
plot_dir = r'/Users/zhengshuyi'

try:
    df = pd.read_csv(ticker_file)
    positive = df[df['Return']>0].groupby(['Year'], as_index=False)['Year'].agg({'Positive Returns': 'count'})
    negative = df[df['Return']<0].groupby(['Year'], as_index=False)['Year'].agg({'Negative Returns': 'count'})
    print(positive)
    print(negative)

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)