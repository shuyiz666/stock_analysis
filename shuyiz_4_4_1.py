'''
assignment4: plot portfolio growth
'''
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn . preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")


def predict(training,Labels,new,distance_parameter_p):
    df_dists = pd.DataFrame(columns=['label', 'distance'])
    labels = []
    for i in new:
        for j in range(len(training)):
            distance = np.linalg.norm(i - training[j], ord=distance_parameter_p)
            df_dists.loc[j] = [Labels[j], distance]
        Sorted_df_dists = df_dists.sort_values(by='distance', ascending=True)
        toplabel = Sorted_df_dists['label'][0:5]
        # freq label, frequency
        predict_label = Counter(toplabel).most_common(1)[0][0]
        labels.append(predict_label)
    labels = labels
    return labels

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
df2017 = df[df['Year'] == 2017]
df2018 = df[df['Year'] == 2018]

training = df2017[['mean_return','volatility']].values
new = df2018[['mean_return','volatility']].values
Labels = df2017['label'].values
true_labes = df2018['label'].values

knn_1 = predict(training,Labels,new,1)
knn_1_5 = predict(training,Labels,new,1.5)
knn_2 = predict(training,Labels,new,2)

# trading with buy-hold
money = 100
flag = 0
portfolio_buy_hold = 100
portfolios_buy_hold = []
i = 0
for index, row in df2018.iterrows():
    # buy hold
    if i == 0:
        portfolios_buy_hold.append(100)
        stock_buy_hold = 100/row['Adj Close']
    else:
        portfolios_buy_hold.append(stock_buy_hold*row['Adj Close'])

# trading with true labels
money = 100
flag = 0
portfolio_true_label = 100
portfolios_true_label = []
i = 0
for index, row in df2018.iterrows():
    # red to green, buy stock
    if true_labes[i] == 'green' and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        portfolio_true_label = shares * row['Adj Close']
    # green to green, do nothing
    elif true_labes[i] == 'green' and flag == 1:
        portfolio_true_label = shares * row['Adj Close']
    # red to red, do nothing
    elif true_labes[i] == 'red' and flag == 0:
        pass
    # green to red, sell stock
    elif true_labes[i] == 'red' and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        portfolio_true_label = money
    i += 1
    portfolios_true_label.append(portfolio_true_label)

# trading with knn1
money = 100
flag = 0
portfolio_knn1 = 100
portfolios_knn1 = []
i = 0
for index, row in df2018.iterrows():
    # red to green, buy stock
    if knn_1[i] == 'green' and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        portfolio_knn1 = shares * row['Adj Close']
    # green to green, do nothing
    elif knn_1[i] == 'green' and flag == 1:
        portfolio_knn1 = shares * row['Adj Close']
    # red to red, do nothing
    elif knn_1[i] == 'red' and flag == 0:
        pass
    # green to red, sell stock
    elif knn_1[i] == 'red' and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        portfolio_knn1 = money
    i += 1
    portfolios_knn1.append(portfolio_knn1)

# trading with knn1.5
money = 100
flag = 0
portfolio_knn1_5 = 100
portfolios_knn1_5 = []
i = 0
for index, row in df2018.iterrows():
    # red to green, buy stock
    if knn_1_5[i] == 'green' and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        portfolio_knn1_5 = shares * row['Adj Close']
    # green to green, do nothing
    elif knn_1_5[i] == 'green' and flag == 1:
        portfolio_knn1_5 = shares * row['Adj Close']
    # red to red, do nothing
    elif knn_1_5[i] == 'red' and flag == 0:
        pass
    # green to red, sell stock
    elif knn_1_5[i] == 'red' and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        portfolio_knn1_5 = money
    i += 1
    portfolios_knn1_5.append(portfolio_knn1_5)

# trading with knn2
money = 100
flag = 0
portfolio_knn2 = 100
portfolios_knn2 = []
i = 0
for index, row in df2018.iterrows():
    # red to green, buy stock
    if knn_2[i] == 'green' and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        portfolio_knn2 = shares * row['Adj Close']
    # green to green, do nothing
    elif knn_2[i] == 'green' and flag == 1:
        portfolio_knn2 = shares * row['Adj Close']
    # red to red, do nothing
    elif knn_2[i] == 'red' and flag == 0:
        pass
    # green to red, sell stock
    elif knn_2[i] == 'red' and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        portfolio_knn2 = money
    i += 1
    portfolios_knn2.append(portfolio_knn2)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.title('5 strategies portfolio in 2018')
plt.xlabel('week_number', fontsize=14)
xlabels = df2018['Week_Number']
plt.ylabel('portfolio', fontsize=14)
plt.yticks()
plt.plot(xlabels,portfolios_buy_hold)
plt.plot(xlabels,portfolios_true_label)
plt.plot(xlabels,portfolios_knn1)
plt.plot(xlabels,portfolios_knn1_5)
plt.plot(xlabels,portfolios_knn2)
plt.legend(['buy hold, mu:'+str(round(np.mean(portfolios_buy_hold),2))+'sigma:'+str(round(np.std(portfolios_buy_hold),2))
            ,'true labels, mu:'+str(round(np.mean(portfolios_true_label),2))+'sigma:'+str(round(np.std(portfolios_true_label),2))
            ,'knn p=1, mu:'+str(round(np.mean(portfolios_knn1),2))+'sigma:'+str(round(np.std(portfolios_knn1),2))
            ,'knn p=1.5, mu:'+str(round(np.mean(portfolios_knn1_5),2))+'sigma:'+str(round(np.std(portfolios_knn1_5),2))
            ,'knn p=2, mu:'+str(round(np.mean(portfolios_knn2),2))+'sigma:'+str(round(np.std(portfolios_knn2),2))])
plt.show()
