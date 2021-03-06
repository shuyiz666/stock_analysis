import os

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    # today's open: row[7]
    # yesterday's close: pre_row[12]
    # today's close: row[12]
    profit = {'long': [], 'short': []}
    for i in range(2,len(lines)):
        row = lines[i].split(',')
        pre_row = lines[i-1].split(',')
        # long position (buy)
        if float(row[7]) > float(pre_row[12]):
            position = 'long'
            daily_profit = round((100/float(row[7]))*(float(row[12])-float(row[7])),2)
        # short position (sell)
        elif float(row[7]) < float(pre_row[12]):
            position = 'short'
            daily_profit = round((100/float(row[7]))*(float(row[7])-float(row[12])),2)
        profit[position].append(daily_profit)
    if len(profit['long']) == 0:
        profit_long = 0
    else:
        profit_long = round(sum(profit['long'])/len(profit['long']),2)
    if len(profit['short']) == 0:
        profit_short = 0
    else:
        profit_short = round(sum(profit['short'])/len(profit['short']),2)
    print('Question 2:')
    print('the profit from "long" positions is: ', profit_long,'%')
    print('the profit from "short" positions is: ', profit_short,'%')
    if sum(profit['long']) > sum(profit['short']):
        print('"long" position is more profitable')
    elif sum(profit['long']) < sum(profit['short']):
        print('"short" position is more profitable')
    elif sum(profit['long']) == sum(profit['short']):
        print('They are equal profitable')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
  
 
