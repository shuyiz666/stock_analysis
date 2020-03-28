'''
assignment2: trading with labels
question2: plot the ”growth” of your account. Week numbers on x, account balance on y
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

for year in range(2017,2019):
    df = pd.read_csv(ticker_file)
    df = df[df['Year'] == year]
    money = 100
    # flag = 0 only have money 1 only have stock
    flag = 0
    value = 100
    values = []
    x = 0
    for index, row in df.iterrows():
        # trading by labels
        # red to green, buy stock
        if row['label'] == 'green' and flag == 0:
            shares = money / row['Adj Close']
            money = 0
            flag = 1
            value = shares * row['Adj Close']
        # green to green, do nothing
        elif row['label'] == 'green' and flag == 1:
            value = shares * row['Adj Close']
        # red to red, do nothing
        elif row['label'] == 'red' and flag == 0:
            value = value
        # green to red, sell stock
        elif row['label'] == 'red' and flag == 1:
            money = shares * row['Adj Close']
            shares = 0
            flag = 0
            value = money
        values.append(value)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    plt.title('growth of balance in '+str(year))
    plt.xlabel('week numbers')
    plt.ylabel('portfolio')
    plt.yticks()
    plt.plot(df['Week_Number'],values)
    plt.show()
 
