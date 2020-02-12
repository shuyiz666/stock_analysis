'''
assignment1: label weeks plot the money hold
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

try:
    df = pd.read_csv(ticker_file)
    money = 100
    # 0 only have money 1 only have stock
    flag = 0
    value = 100
    values = []
    values_hold = []
    # red green 买 有股票没钱
    # green red 卖 有钱没股票
    # green green 不动 有股票没钱
    # red red 不动 有钱没股票
    x = 0
    for index, row in df.iterrows():
        # buy hold strategy
        if x == 0:
            values_hold.append(100)
            x += 1
            stock = 100/row['Adj Close']
        else:
            values_hold.append(stock*row['Adj Close'])
        # red to green, buy stock
        if row['label'] == 'green' and flag == 0:
            shares = money/row['Adj Close']
            money = 0
            flag = 1
            value = shares*row['Adj Close']
        # green to green, do nothing
        elif row['label'] == 'green' and flag == 1:
            value = shares*row['Adj Close']
        # red to red, do nothing
        elif row['label'] == 'red' and flag == 0:
            value = shares*row['Adj Close']
        # green to red, sell stock
        elif row['label'] == 'green' and flag == 1:
            money = shares/row['Adj Close']
            shares = 0
            value = money
        values.append(value)
    plt.figure(figsize=(10, 5))
    plt.title('')
    plt.xlabel('date', fontsize=14)
    plt.ylabel('value', fontsize=14)
    plt.plot(df['Date'],values)
    plt.plot(df['Date'],values_hold)
    plt.legend(['trading by label','buy_hold'])
    plt.show()




except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)