'''
assignment2: Weekly Trading with Linear Models
question4: implement three trading strategies for year 2 (for each d using the ”best” values for W from year 1 that you have computed)
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def linear_model(W,d,df):
    index_year2 = df[df['Year'] == 2018].index.values.astype(int)[0]
    x = df['Week_Number'].values
    y = df['Adj Close'].values
    start = index_year2-W
    end = index_year2
    results, real_results = [], []
    while end <= df.index[-1]:
        train_x = x[start:end] # week number
        train_y = y[start:end] # close price
        testing_x = x[end]
        weights = np.polyfit(train_x,train_y,d)
        model = np.poly1d(weights)
        predicted = model(testing_x)
        if predicted > y[end-1]:
            results.append('green')
        elif predicted < y[end-1]:
            results.append('red')
        else:
            results.append(df.loc[end-1,'label'])
        start += 1
        end += 1
    return results

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
index_year2 = df[df['Year']==2018].index.values.astype(int)
real_label = df[df['Year']==2018]['label'].values
testing  = df[df['Year']==2018]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.title('portfolio in three strategies with different d and W')
plt.xlabel('week numbers')
plt.ylabel('portfolio')
plt.yticks()

predicted_labels = []
predicted_labels.append(linear_model(12, 1, df))
predicted_labels.append(linear_model(9, 2, df))
predicted_labels.append(linear_model(11, 3, df))

for results in predicted_labels:
    # trading by predict_labels
    money = 100
    # flag = 0 only have money 1 only have stock
    flag = 0
    portfolio = 100
    portfolios = []
    i = 0
    for index, row in testing.iterrows():
        # trading with labels
        # red to green, buy stock
        if results[i] == 'green' and flag == 0:
            shares = money / row['Adj Close']
            money = 0
            flag = 1
            portfolio = shares * row['Adj Close']
        # green to green, do nothing
        elif results[i] == 'green' and flag == 1:
            portfolio = shares * row['Adj Close']
        # red to red, do nothing
        elif results[i] == 'red' and flag == 0:
            pass
        # green to red, sell stock
        elif results[i] == 'red' and flag == 1:
            money = shares * row['Adj Close']
            shares = 0
            flag = 0
            portfolio = money
        i += 1
        portfolios.append(portfolio)
    plt.plot(testing['Week_Number'], portfolios)

plt.legend(['d=1,W=12','d=21,W=9','d=3,W=11'])
plt.show()


