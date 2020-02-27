'''
assignment3: Linear Separability
question3: implement a trading strategy based on your labels for year 2
'''
import os
import pandas as pd
import matplotlib.pyplot as plt

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

for index, row in df.iterrows():
    # trading by labels
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
    elif row['label'] == 'green' and flag == 1:
        money = shares*row['Adj Close']
        shares = 0
        value = money
    plt.title('growth of balance in '+str(year))
    plt.xlabel('week numbers')
    plt.ylabel('stock holding price')
    plt.plot(df['Week_Number'],values)
    plt.show()
