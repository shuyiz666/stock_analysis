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
    df = df[df['Year']==2017]
    money = 100
    shares = 0
    # flag = 0 only have money 1 only have stock
    flag = 0
    value = 100
    values = []
    values_hold = []
    x = 0
    for index, row in df.iterrows():
        # buy hold strategy
        if x == 0:
            values_hold.append(100)
            x += 1
            stock = 100/row['Adj Close']
        else:
            values_hold.append(stock*row['Adj Close'])

        # trading by labels
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
            value = value
        # green to red, sell stock
        elif row['label'] == 'red' and flag == 1:
            money = shares*row['Adj Close']
            shares = 0
            flag = 0
            value = money
        values.append(value)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    plt.title('comparison for two strategy')
    plt.xlabel('week_number', fontsize=14)
    xlabels = df['Week_Number']
    ax.axes.set_xticklabels(xlabels, rotation=90,fontsize=5)
    plt.ylabel('stock holding price', fontsize=14)
    plt.plot(xlabels,values)
    plt.plot(df['Year'].map(str)+'/'+df['Week_Number'].map(str),values_hold)
    plt.legend(['trading by label','buy_hold'])
    plt.show()

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
    
