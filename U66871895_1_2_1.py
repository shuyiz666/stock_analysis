import os

ticker = 'ZSAN'
input_dir = r'/Users/zhengshuyi'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    # today's open: row[7]
    # yesterday's close: pre_row[12]
    # today's close: row[12]
    profit = []
    for i in range(2,len(lines)):
        row = lines[i].split(',')
        pre_row = lines[i-1].split(',')
        # long position (buy)
        if float(row[7]) > float(pre_row[12]):
            daily_profit = round((100/float(row[7]))*(float(row[12])-float(row[7])),2)
        # short position (sell)
        elif float(row[7]) < float(pre_row[12]):
            daily_profit = round((100/float(row[7]))*(float(row[7])-float(row[12])),2)
        # equal (no trade)
        else:
            daily_profit = 0
        profit.append(daily_profit)
    avg_profit = round(sum(profit)/len(profit),2)
    print('Question 1: the average daily profit is ', avg_profit,'%')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
