import os
import matplotlib.pyplot as plt

ticker = 'ZSAN'
input_dir = r'/Users/zhengshuyi'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    # data in different restrictions
    xlabel, ylabel_long,ylabel_short = [], [], []
    # today's open: row[7]
    # yesterday's close: pre_row[12]
    # today's close: row[12]
    for j in range(100):
        profit = {'long': [], 'short': [], 'equal': []}
        v = round(j*0.001,3)
        for i in range(2,len(lines)):
            row = lines[i].split(',')
            pre_row = lines[i-1].split(',')
            # long position (buy)
            if float(row[7]) > float(pre_row[12])*(v+1.00):
                position = 'long'
                daily_profit = (100/float(row[7]))*(float(row[12])-float(row[7]))
            # short position (sell)
            elif float(row[7]) < float(pre_row[12])*(1.00-v):
                position = 'short'
                daily_profit = (100/float(row[7]))*(float(row[7])-float(row[12]))
            # equal (no trade)
            else:
                position = 'equal'
                daily_profit = 0
            profit[position].append(daily_profit)
        avg_profit = round((sum(profit['equal']) + sum(profit['long']) + sum(profit['short']))/(len(lines)-2),2)
        xlabel.append(v)
        if len(profit['long']) == 0:
            avg_long = 0
        else:
            avg_long = round(sum(profit['long'])/len(profit['long']),2)
        if len(profit['short']) == 0:
            avg_short = 0
        else:
            avg_short = round(sum(profit['short'])/len(profit['short']),2)
        ylabel_long.append(avg_long)
        ylabel_short.append(avg_short)

    print('Question 4:')
    print('In short position:')
    print('The profit overall decline but fluctuate slightly.')
    print('The lowest point is when x from 9.1% to 9.3% and it loss 7.34% money.')
    print('')
    print('In long position:')
    print('The pattern is similar to the short position.But the The curve goes down more gently.')

    # show short position plot
    plt.plot(xlabel,ylabel_short)
    plt.xlabel('threshold value')
    plt.ylabel('Average Profit')
    plt.title('The average profit per trade in "short" positions')
    plt.show()

    # show long position plot
    plt.plot(xlabel,ylabel_long)
    plt.xlabel('threshold value')
    plt.ylabel('Average Profit')
    plt.title('The average profit per trade in "long" positions')
    plt.show()

    # show long position and short position plot
    plt.plot(xlabel,ylabel_long,label = 'long')
    plt.plot(xlabel,ylabel_short,label = 'short')
    plt.xlabel('threshold value')
    plt.ylabel('Average Profit')
    plt.title('The average profit per trade')
    plt.legend()
    plt.show()

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
