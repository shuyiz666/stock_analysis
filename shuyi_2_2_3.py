'''
assigment 2: Bollinger bands
question 3: repeat the previous question for 2018
'''
import os
import pandas as pd
import matplotlib.pylab as plt
import timeit

start = timeit.default_timer()


ticker = 'ZSAN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_weekly_return_volatility.csv')
df = pd.read_csv(ticker_file)
df2017 = df[df['Year'] == 2018]


def bollinger(w,k):
    position = 'no'
    shares = 0
    profit = 0
    transaction = 0
    for index, row in df2017.iterrows():
        # average close in w period
        ma = df[index:index+w]['Adj Close'].mean()
        upper = ma+k*row['volatility']
        lower = ma-k*row['volatility']
        P = row['Adj Close']
        if P > upper:
            if position == 'no':
                shares += 100/P
                position = 'short'
            elif position == 'long':
                profit += shares*P-100
                shares = 0
                position = 'no'
                transaction += 1
        elif P < lower:
            if position == 'no':
                shares += 100/P
                position = 'long'
            elif position == 'short':
                profit += 100-shares*P
                shares = 0
                position = 'no'
                transaction += 1
    if transaction == 0:
        avg_profit = 0
    else:
        avg_profit = profit/transaction
    return avg_profit

try:
    df = pd.read_csv(ticker_file)
    df2017 = df[df['Year'] == 2018]
    k_range = [0.5,1,1.5,2,2.5]
    scatter_positive = pd.DataFrame(columns=('w','k','value'))
    scatter_negative = pd.DataFrame(columns=('w','k','value'))
    row_positive, row_negative = 0, 0
    for w in range(10,51):
        for k in k_range:
            avg_p = bollinger(w,k)
            print('w = ',w,', k = ',k,', avg_profit = ',avg_p)
            if avg_p > 0:
                scatter_positive.loc[row_positive] = {'w':w,'k':k,'value':avg_p}
                row_positive += 1
            elif avg_p < 0:
                scatter_negative.loc[row_negative] = {'w':w,'k':k,'value':avg_p}
                row_negative += 1
    plt.figure(figsize = (10,5))
    plt.title('Average P/L per transaction with change of W nad k')
    plt.xlabel('W', fontsize = 14)
    plt.ylabel('k', fontsize = 14)
    plt.scatter(scatter_positive['w'],scatter_positive['k'],s=scatter_positive['value'],c='green')
    plt.scatter(scatter_negative['w'],scatter_negative['k'],s=-scatter_negative['value'],c='red')
    plt.legend(['profit','loss'])
    plt.show()
    print('It seems when k becomes bigger, the profit becomes higher.')
    print('When k = 1.2 and k = 0.8, there is point which means there is no transaction happened.')
    print('The transactions are less than 2017.')

except Exception as e:
    # print(e)
    print('failed to read stock data for ticker: ', ticker)