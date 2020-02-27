'''
assignment0: Take your points , take year 1 Your plot is (r,sigma).  Remove a few points so that your weeks are “linearly” separable
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as mtick

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

# plot before removing points in year1
df = pd.read_csv(ticker_file)
df1 = df[df['Year']==2017]
plt.scatter(df1['mean_return'], df1['volatility'],color=df1['label'], s=20,alpha=0.5)
plt.title('plot before removing points')
plt.xlabel('r')
plt.ylabel('sigma')
for a,b,c in zip(df1['mean_return'],df1['volatility'],df1['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show()

# plot after removing points in year 1
df_remove = df1[~df1.Week_Number.isin([4,12,21,22])]
plt.scatter(df_remove['mean_return'], df_remove['volatility'],color=df_remove['label'], s=20,alpha=0.5)
plt.plot([1,2.2],[0,18],color = 'black', ls = 'dotted')
plt.title('plot after removing points')
plt.xlabel('r')
plt.ylabel('sigma')
for a,b,c in zip(df_remove['mean_return'],df_remove['volatility'],df_remove['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show()

# compute the line that separates good and bad weeks
print('the equation is: y = 15*x-15\n')

# take year2 and use this line from year 1 to predict your labels
df2 = df[df['Year']==2018].copy()
df2['predict_label'] = 'NULL'
for index, row in df2.iterrows():
    if row['mean_return']*15-15 >= row['volatility']:
        df2.loc[index,'predict_label'] = 'green'
    else:
        df2.loc[index,'predict_label'] = 'red'

# Compute accuracy and confusion matrix
cm = confusion_matrix(df2['label'], df2['predict_label'])
print('confusion matrix is:\n',cm,'\n')
accuracy = sum(df2['label']==df2['predict_label'])/len(df2['label'])
print('accuracy is: ','%s%%'%(round(accuracy*100,2)))

# predict good/bad weeks plot in year2
plt.scatter(df2['mean_return'], df2['volatility'],color=df2['predict_label'], s=20,alpha=0.5)
plt.plot([1,11/3],[0,40],color = 'black', ls = 'dotted')
plt.title('predict good/bad weeks plot in year2')
plt.xlabel('r')
plt.ylabel('sigma')
for a,b,c in zip(df2['mean_return'],df2['volatility'],df2['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show()

# trading with predict labels in year2
money = 100
# flag = 0 only have money 1 only have stock
flag = 0
value = 100
values_hold = []
x = 0
for index, row in df2.iterrows():
    # red to green, buy stock
    if row['predict_label'] == 'green' and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        value = shares * row['Adj Close']
    # green to green, do nothing
    elif row['predict_label'] == 'green' and flag == 1:
        value = shares * row['Adj Close']
    # red to red, do nothing
    elif row['predict_label'] == 'red' and flag == 0:
        value = value
    # green to red, sell stock
    elif row['predict_label'] == 'red' and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        value = money
    values_hold.append(value)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.title('trading with predict labels in 2018')
plt.xlabel('week_number', fontsize=14)
xlabels = df2['Week_Number']
plt.ylabel('portfolio', fontsize=14)
plt.yticks()
plt.plot(xlabels,values_hold)
plt.show()
