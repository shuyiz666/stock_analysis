import os
# just use this for plot display
import matplotlib.pyplot as plt

ticker = 'ZSAN'
input_dir = r'/Users/zhengshuyi'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    # data in different restrictions
    xlabel, ylabel_total, ylabel_long,ylabel_short = [], [], [], []
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
        ylabel_total.append(avg_profit)
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
        if j == 0:
            print('Question 1: the average daily profit is ', avg_profit)
            if avg_long > avg_short:
                print('Question 2: long position is more profitable')
            elif avg_long < avg_short:
                print('Question 2: short position is more profitable')
            else:
                print('Question 2: they are equal')
        # print(v,avg_profit,avg_long +avg_short,avg_long,avg_short)
    # show total plot
    plt.plot(xlabel,ylabel_total)
    plt.xlabel('threshold value')
    plt.ylabel('Average Profit')
    plt.title('The average profit per trade')
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
