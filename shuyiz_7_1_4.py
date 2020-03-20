'''
assignment1: Day Trading with Linear Regression
question4: what is the average profit/loss per ”long position” trade and per ”short position” trades in year 2?
'''

import os
import pandas as pd
import numpy as np

def linear_model(W,df):
    t = 1 # day
    p = df['Adj Close'].values # price
    start = 0
    end = W

    position = 'no'
    shares = 0
    profit_long = 0
    profit_short = 0
    long = 0
    short = 0

    while end < len(df):
        train_x = np.array(range(t,t+W)) # training days
        train_y = p[start:end] # training prices
        testing_x = t+W # testing day
        weights = np.polyfit(train_x,train_y,1)
        model = np.poly1d(weights)
        predicted = model(testing_x)

        if predicted > p[end-1]:
            if position == 'no':
                shares = 100/p[end-1]
                px = p[end-1]
                position = 'long'


            elif position == 'long':
                long += 1
                pass

            elif position == 'short':
                short += 1
                profit_short += shares*(px-p[end-1])
                shares = 0
                position = 'no'

        elif predicted < p[end-1]:
            if position == 'no':
                shares = 100/p[end-1] # sell short对比丢
                px = p[end-1]
                position = 'short'

            elif position == 'short':
                short += 1
                pass

            elif position == 'long':
                long += 1
                profit_long += shares*(p[end-1]-px)
                shares = 0
                position = 'no'

        elif predicted == p[end-1]:
            pass

        start += 1
        end += 1
        t += 1

    return profit_short/short,profit_long/long

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)
df2018 = df[df['Year']==2018]

short,long = linear_model(5,df2018)

print('profit/loss per ”long position” trade:',round(long,2))
print('profit/loss per ”short position” trade:', round(short,2))
