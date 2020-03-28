'''
assignment1: Day Trading with Linear Regression
question2: use the value of W ∗ from year 1 and consider year 2. For every day in year 2, take the previous W∗ days, compute linear regression and compute the value of r2 for that day. Plot the graph of r2 for year 2. What is the average r2. How well does it explain price movements?
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def linear_model(W,df):
    t = 1 # day
    p = df['Adj Close'].values # price
    start = 0
    end = W
    r2s = []

    while end < len(df):
        train_x = np.array(range(t,t+W)) # training days
        train_y = p[start:end] # training prices
        weights = np.polyfit(train_x,train_y,1)
        model = np.poly1d(weights)
        predicted = model(train_x)
        r2 = r2_score(train_y, predicted)
        r2s.append(r2)
        start += 1
        end += 1
        t += 1

    return r2s

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)
df2018 = df[df['Year']==2018]

plt.title('r square for year 2')
plt.ylabel('r square')

r2s = linear_model(5,df2018)
x = list(range(1,len(r2s)+1))
plt.plot(x,r2s)
plt.show()

print('the average of r square is:', round(np.mean(r2s),2))
print('the r square vary a lot which means the close price of the stock is unstable')

 
