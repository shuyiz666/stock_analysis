'''
assignment1: Day Trading with Linear Regression
question5: what is the average number of days for long position and short position transactions in year 2?
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
    long = 0
    short = 0
    longs = []
    shorts = []


    while end < len(df):
        train_x = np.array(range(t,t+W)) # training days
        train_y = p[start:end] # training prices
        testing_x = t+W # testing day
        weights = np.polyfit(train_x,train_y,1)
        model = np.poly1d(weights)
        predicted = model(testing_x)

        if predicted > p[end-1]:
            if position == 'no':
                long = 0
                short = 0
                position = 'long'

            elif position == 'long':
                long += 1
                pass

            elif position == 'short':
                short += 1
                position = 'no'
                shorts.append(short)

        elif predicted < p[end-1]:
            if position == 'no':
                long = 0
                short = 0
                position = 'short'

            elif position == 'short':
                short += 1
                pass

            elif position == 'long':
                long += 1
                position = 'no'
                longs.append(long)

        elif predicted == p[end-1]:
            pass

        start += 1
        end += 1
        t += 1

    return shorts,longs

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)
df2018 = df[df['Year']==2018]

shorts,longs = linear_model(5,df2018)

print('the average number of days for short position is:',round(np.mean(shorts),2))
print('the average number of days for long position is:', round(np.mean(longs)),2)
