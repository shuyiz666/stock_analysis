'''
assignment2: trading with labels
question5: what was the maximum duration (in weeks) that your ac- count was growing and what was the maximum duration (in weeks) that your account was decreasing in value?
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
    x = 0
    dp = 1
    prevalue = 100
    dur_incre,dur_decre = 1,1
    dp_incre, dp_decre = 1,1
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
        if value > prevalue:
            dur_incre += 1
            dur_decre = 1
        elif value < prevalue:
            dur_decre += 1
            dur_incre = 1
        prevalue = value
        dp_incre = max(dp_incre, dur_incre)
        dp_decre = max(dp_decre, dur_decre)
    print('In ',str(year),':')
    print('the maximum duration of growing is', dp_incre)
    print('the maximum duration of decreasing is',dp_decre,'\n')