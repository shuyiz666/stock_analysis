'''
assignment1: Day Trading with Linear Regression
question6: are these results very different from those in year 1 for this value of W∗
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
df2017 = df[df['Year']==2017]

shorts,longs = linear_model(5,df2017)

print('the average number of days for short position is:',round(np.mean(shorts),2))
print('the average number of days for long position is:', round(np.mean(longs)),2)
print('the average number of days for short position in year1 is longer than year2')
print('the average number of days for longer position in year1 is shorter than year2')