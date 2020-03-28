'''
assignment2: trading with labels
question1: What is the average and volatility of weekly balances
'''

import os
import pandas as pd
import statistics as st

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

df = pd.read_csv(ticker_file)
money = 100
# flag = 0 only have money 1 only have stock
flag = 0
value = 100
values2017,values2018 = [],[]
values_hold = []
x = 0
for index, row in df.iterrows():
    # trading by labels
    # red to green, buy stock
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
    if row['Year'] == 2017:
        values2017.append(value)
    elif row['Year'] == 2018:
        values2018.append(value)

print('2017: ')
print('the average of weekly balances is ',round(st.mean(values2017),2))
print('the volatility of weekly balances is ',round(st.stdev(values2017),2),'\n')
print('2018: ')
print('the average of weekly balances is ',round(st.mean(values2018),2))
print('the volatility of weekly balances is ',round(st.stdev(values2018),2))
