'''
assignment2: naive bayesian
question4: implement a trading strategy based on your labels for year 2 and compare the performance with the ”buy-and-hold” strategy. Which strategy results in a larger amount at the end of the year?
'''
import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
traning = df[df['Year']==2017]
testing = df[df['Year']==2018]

X = traning[['mean_return','volatility']].values
Y = traning[['label']].values

NB_classifier = GaussianNB().fit(X,Y.ravel())

new_instance = np.asmatrix(testing[['mean_return','volatility']].values)
prediction = NB_classifier.predict(new_instance)



# trading by predict_labels
money = 100
# flag = 0 only have money 1 only have stock
flag = 0
portfolio = 100
portfolios,portfolios_buy_hold = [],[]
i = 0
for index, row in testing.iterrows():
    # buy hold
    if i == 0:
        portfolios_buy_hold.append(100)
        stock_buy_hold = 100/row['Adj Close']
    else:
        portfolios_buy_hold.append(stock_buy_hold*row['Adj Close'])

    # trading with labels
    # red to green, buy stock
    if prediction[i] == 'green' and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        portfolio = shares * row['Adj Close']
    # green to green, do nothing
    elif prediction[i] == 'green' and flag == 1:
        portfolio = shares * row['Adj Close']
    # red to red, do nothing
    elif prediction[i] == 'red' and flag == 0:
        pass
    # green to red, sell stock
    elif prediction[i] == 'red' and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        portfolio = money
    i += 1
    portfolios.append(portfolio)

if portfolios[-1] > portfolios_buy_hold[-1]:
    print('trading strategy based on naive bayesian predicted labels results in a larger amount at the end of the year')
elif portfolios[-1] == portfolios_buy_hold[-1]:
    print('trading strategy based on naive bayesian predicted labels results the same amount with buy hold strategy at the end of the year')
else:
    print('buy-and-hold results in a larger amount at the end of the year')

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.title('two strategies portfolio in 2018')
plt.xlabel('week_number', fontsize=14)
xlabels = testing['Week_Number']
plt.ylabel('portfolio', fontsize=14)
plt.yticks()
plt.plot(xlabels,portfolios)
plt.plot(xlabels,portfolios_buy_hold)
plt.legend(['trading with labels (naive bayesian)','buy hold'])
plt.show() 

