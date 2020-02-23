'''
assignment4: plot portfolio growth
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import statistics as st

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
data = df[df['Year']==2018]

def buy_hold():
    portfolios = []
    x = 0
    for index, row in data.iterrows():
        # buy hold strategy
        if x == 0:
            portfolios.append(100)
            x += 1
            stock = 100 / row['Adj Close']
        else:
            portfolios.append(stock * row['Adj Close'])
    avg = st.mean(portfolios)
    std = st.stdev(portfolios)
    return portfolios, avg,std

def true_label():
    money = 100
    shares = 0
    # flag = 0 only have money 1 only have stock
    flag = 0
    portfolio = 100
    portfolios = []
    for index, row in data.iterrows():
        # red to green, buy stock
        if row['label'] == 'green' and flag == 0:
            shares = money/row['Adj Close']
            money = 0
            flag = 1
            portfolio = shares*row['Adj Close']
        # green to green, do nothing
        elif row['label'] == 'green' and flag == 1:
            portfolio = shares*row['Adj Close']
        # red to red, do nothing
        elif row['label'] == 'red' and flag == 0:
            portfolio = portfolio
        # green to red, sell stock
        elif row['label'] == 'red' and flag == 1:
            money = shares*row['Adj Close']
            shares = 0
            flag = 0
            portfolio = money
        portfolios.append(portfolio)
    avg = st.mean(portfolios)
    std = st.stdev(portfolios)
    return portfolios, avg, std

def KNN():
    training = df[df['Year'] == 2017]
    X = training[['mean_return', 'volatility']].values
    Y = training[['label']].values
    avgs = {}
    stds = {}
    portfolio_dic = {}
    ks = [1, 1.5, 2]
    for k in ks:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X, np.ravel(Y))
        new_instance = data[['mean_return', 'volatility']].values
        prediction = knn_classifier.predict(new_instance)
        i = 0
        money = 100
        shares = 0
        # flag = 0 only have money 1 only have stock
        flag = 0
        portfolio = 100
        portfolios = []
        for index, row in data.iterrows():
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
                portfolio = portfolio
            # green to red, sell stock
            elif prediction[i] == 'red' and flag == 1:
                money = shares * row['Adj Close']
                shares = 0
                flag = 0
                portfolio = money
            i += 1
            portfolios.append(portfolio)
        print(k)
        avgs[k] = st.mean(portfolios)
        stds[k] = st.stdev(portfolios)
        portfolio_dic[k] = portfolios
        return portfolio_dic, avgs, stds

buy_hold, buy_hold_avg, buy_hold_std = buy_hold()
true_label, true_label_avg, true_label_std = true_label()
knn, knn_avg, knn_std = KNN()
print(knn)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
plt.title('portfolio growth')
plt.xlabel('week_number', fontsize=14)
xlabels = data['Week_Number']
ax.axes.set_xticklabels(xlabels, rotation=90,fontsize=5)
plt.ylabel('portfolio growth', fontsize=14)
plt.plot(xlabels,buy_hold)
plt.plot(xlabels,true_label)
plt.plot(xlabels,knn[1])
plt.plot(xlabels,knn[1.5])
plt.plot(xlabels,knn[2])
plt.legend(['buy_hold, mean = '+str(round(buy_hold_avg,2))+', std = '+str(round(buy_hold_std,2)),
            'true_label, mean = '+str(round(true_label_avg,2))+', std = '+str(round(true_label_std,2)),
            'knn: k = 1, mean = '+str(round(knn_avg[1],2))+', std = '+str(round(knn_std[1],2)),
            'knn: k = 1.5, mean = '+str(round(knn_avg[1.5],2))+', std = '+str(round(knn_std[1.5],2)),
            'knn: k = 2, mean = '+str(round(knn_avg[2],2))+', std = '+str(round(knn_std[2],2))])
plt.show()