'''
assignment1: Day Trading with Linear Regression
question1: take W = 5,6,...,30 and consider your data for year 1. For each W in the specified range, compute your average P/L per trade and plot it: on x-axis you plot the values of W and on the y axis you plot profit and loss per trade. What is the optimal value W∗ of W?
'''
import os
import pandas as pd
import numpy as np

def linear_model(W,df):
    index_year2 = df[df['Year'] == 2017].index.values.astype(int)[0]
    t = 1 # day
    p = df['Adj Close'].values # price
    start = index_year2-W
    end = index_year2

    position = 'no'
    shares = 0
    trade = 0
    profit = 0
    while end <= df.index[-1]:
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
                pass

            elif position == 'short':
                profit += shares*(px-p[end-1])
                trade += 1
                shares = 0
                position = 'no'

        elif predicted < p[end-1]:
            if position == 'no':
                shares = 100/p[end-1] # sell short对比丢
                px = p[end-1]
                position = 'short'

            elif position == 'short':
                pass

            elif position == 'long':
                profit += shares*(p[end-1]-px)
                trade += 1
                shares = 0
                position = 'no'

        elif predicted == p[end-1]:
            pass

        start += 1
        end += 1
        t += 1
    profit_per_trade = profit / trade
    return profit_per_trade

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)

for W in range(5,10):
    profit_per_trade = linear_model(W,df)
    print(profit_per_trade)
