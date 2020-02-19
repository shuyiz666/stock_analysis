'''
assignment2: trading with labels
question3: what (and when) was the min & max of the account?
'''
import os
import pandas as pd

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
    values = {}
    x = 0
    for index, row in df.iterrows():
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
            value = shares*row['Adj Close']
        # green to red, sell stock
        elif row['label'] == 'green' and flag == 1:
            money = shares*row['Adj Close']
            shares = 0
            value = money
        values[row['Week_Number']] = value
    print('in',str(year),':')
    print('when the week number is', max(values, key=values.get), ', the balance reached its peak at',round(max(values.values()),2))
    print('when the week number is', min(values, key=values.get), ', the balance became least at',round(min(values.values()), 2),'\n')
