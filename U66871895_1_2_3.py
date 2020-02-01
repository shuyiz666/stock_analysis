import os
import matplotlib.pyplot as plt

ticker = 'ZSAN'
input_dir = r'/Users/zhengshuyi'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    # data in different restrictions
    xlabel, ylabel= [], []
    # today's open: row[7]
    # yesterday's close: pre_row[12]
    # today's close: row[12]
    for j in range(100):
        profit = []
        v = round(j*0.001,3)
        for i in range(2,len(lines)):
            row = lines[i].split(',')
            pre_row = lines[i-1].split(',')
            # long position (buy)
            if float(row[7]) > float(pre_row[12])*(v+1.00):
                daily_profit = (100/float(row[7]))*(float(row[12])-float(row[7]))
            # short position (sell)
            elif float(row[7]) < float(pre_row[12])*(1.00-v):
                daily_profit = (100/float(row[7]))*(float(row[7])-float(row[12]))
            # equal (no trade)
            else:
                daily_profit = 0
            profit.append(daily_profit)
        avg_profit = round(sum(profit)/len(profit),2)
        xlabel.append(v)
        ylabel.append(avg_profit)
    # show total plot
    print('Question 3:')
    print('The general average profit trend is going up with the threshold value become bigger.')
    print('But, The average profit goes down when the value from 0 to 1.15%.')
    print('The optimal values for x is bigger than 9.7%, the profit is -0.12%.')
    print('The worst case is when threshold value = 1.3%, at that point, the profit is -0.49%.')
    plt.plot(xlabel,ylabel)
    plt.xlabel('threshold value')
    plt.ylabel('Average Profit')
    plt.title('The average profit per trade')
    plt.show()

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
