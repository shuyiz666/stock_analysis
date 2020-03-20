'''
assignment1: Day Trading with Linear Regression
question1: take W = 5,6,...,30 and consider your data for year 1. For each W in the specified range, compute your average P/L per trade and plot it: on x-axis you plot the values of W and on the y axis you plot profit and loss per trade. What is the optimal value W∗ of W?
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def linear_model(W,df):
    t = 1 # day
    p = df['Adj Close'].values # price
    start = 0
    end = W

    position = 'no'
    shares = 0
    profit = 0
    transaction = 0
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
                pass

            elif position == 'short':
                profit += shares*(px-p[end-1])
                shares = 0
                position = 'no'
                transaction += 1

        elif predicted < p[end-1]:
            if position == 'no':
                shares = 100/p[end-1] # sell short对比丢
                px = p[end-1]
                position = 'short'

            elif position == 'short':
                pass

            elif position == 'long':
                profit += shares*(p[end-1]-px)
                shares = 0
                position = 'no'
                transaction += 1

        elif predicted == p[end-1]:
            pass

        start += 1
        end += 1
        t += 1
    return profit/transaction

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)
df2017 = df[df['Year']==2017]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fmt = '${x:,.2f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.title('Profit/Loss per trade with different W')
plt.xlabel('W')
plt.ylabel('P/L')
plt.yticks()


Ws = list(range(5,31))
profits = []
for W in Ws:
    profit_per_trade = linear_model(W,df2017)
    profits.append(profit_per_trade)

plt.plot(Ws, profits)
plt.show()

print('the optimal value of W is:', Ws[profits.index(max(profits))])
