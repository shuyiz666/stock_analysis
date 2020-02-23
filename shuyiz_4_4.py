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
    values_hold = []
    x = 0
    for index, row in data.iterrows():
        # buy hold strategy
        if x == 0:
            values_hold.append(100)
            x += 1
            stock = 100 / row['Adj Close']
        else:
            values_hold.append(stock * row['Adj Close'])
    avg = st.mean(values_hold)
    std = st.stdev(values_hold)
    return values_hold, avg,std

def true_label():
    money = 100
    # flag = 0 only have money 1 only have stock
    flag = 0
    value = 100
    values = []
    for index, row in data.iterrows():
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
        elif row['label'] == 'red' and flag == 1:
            money = shares*row['Adj Close']
            shares = 0
            value = money
        values.append(value)
    avg = st.mean(values)
    std = st.stdev(values)
    return values, avg, std

def KNN():
    training = df[df['Year'] == 2017]
    X = training[['mean_return', 'volatility']].values
    Y = training[['label']].values
    avgs = {}
    stds = {}
    values_dic = {}
    ks = [1, 1.5, 2]
    for k in ks:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X, np.ravel(Y))
        new_instance = data[['mean_return', 'volatility']].values
        prediction = knn_classifier.predict(new_instance)
        i = 0
        money = 100
        # flag = 0 only have money 1 only have stock
        flag = 0
        value = 100
        values = []
        for index, row in data.iterrows():
            # red to green, buy stock
            if prediction[i] == 'green' and flag == 0:
                shares = money / row['Adj Close']
                money = 0
                flag = 1
                value = shares * row['Adj Close']
            # green to green, do nothing
            elif prediction[i] == 'green' and flag == 1:
                value = shares * row['Adj Close']
            # red to red, do nothing
            elif prediction[i] == 'red' and flag == 0:
                value = shares * row['Adj Close']
            # green to red, sell stock
            elif prediction[i] == 'red' and flag == 1:
                money = shares * row['Adj Close']
                shares = 0
                value = money
            i += 1
            values.append(value)
        avgs[k] = st.mean(values)
        stds[k] = st.stdev(values)
        values_dic[k] = values
        return values_dic, avgs, stds

print(buy_hold())
print(true_label())
print(KNN())




    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(111)
    # plt.title('comparison for two strategy')
    # plt.xlabel('day', fontsize=14)
    # xlabels = df['Year'].map(str)+'/'+df['Week_Number'].map(str)
    # ax.axes.set_xticklabels(xlabels, rotation=90,fontsize=5)
    # plt.ylabel('stock holding price', fontsize=14)
    # plt.plot(xlabels,values)
    # plt.plot(df['Year'].map(str)+'/'+df['Week_Number'].map(str),values_hold)
    # plt.legend(['trading by label','buy_hold'])
    # plt.show()