'''
assignment3: Linear Separability
question3: implement a trading strategy based on your labels for year 2
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

df = pd.read_csv(ticker_file)
df = df[df['Year']==2018]
df['predict_label'] = 'NULL'

# assign year2
for index, row in df.iterrows():
    if row['mean_return']*15-15 >= row['volatility']:
        df.loc[index,'predict_label'] = 'green'
    else:
        df.loc[index,'predict_label'] = 'red'

# trading by predict_labels
money = 100
# flag = 0 only have money 1 only have stock
flag = 0
value = 100
values_hold = []
x = 0
for index, row in df.iterrows():
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
xlabels = df['Week_Number']
plt.ylabel('portfolio', fontsize=14)
plt.yticks()
plt.plot(xlabels,values_hold)
plt.show() 
