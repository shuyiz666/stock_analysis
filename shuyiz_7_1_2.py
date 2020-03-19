'''
assignment1: Day Trading with Linear Regression
question2: use the value of W ∗ from year 1 and consider year 2. For every day in year 2, take the previous W∗ days, compute linear regression and compute the value of r2 for that day. Plot the graph of r2 for year 2. What is the average r2. How well does it explain price movements?
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import r2_score

def linear_model(W,df):
    index_year2 = df[df['Year'] == 2018].index.values.astype(int)[0]
    t = 1 # day
    p = df['Adj Close'].values # price
    start = index_year2-W
    end = index_year2
    r2s = []
    while end <= df.index[-1]:
        train_x = np.array(range(t,t+W)) # training days
        train_y = p[start:end] # training prices
        testing_x = t+W # testing day
        weights = np.polyfit(train_x,train_y,1)
        model = np.poly1d(weights)
        predicted = model(testing_x)
        print(train_y,predicted)
        r2 = r2_score(train_y, predicted)
        print(r2)
        start += 1
        end += 1
        t += 1
        print(t)

    return r2s

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.title('Profit/Loss per trade with different W')
plt.xlabel('W')
plt.ylabel('P/L')
plt.yticks()

r2s = linear_model(5,df)
# plt.plot(x,r2s)
# plt.show()


