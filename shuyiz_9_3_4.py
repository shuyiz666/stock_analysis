'''
assignment3: random forest
question4: implement a trading strategy based on your labels for year 2 and compare the performance with the ”buy-and-hold” strategy. Which strategy results in a larger amount at the end of the year
'''
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
traning = df[df['Year']==2017]
testing = df[df['Year']==2018]

def random_forest(traning, testing, d, N):
    X = traning[['mean_return', 'volatility']].values
    le = LabelEncoder()
    Y = le.fit_transform(traning['label'].values)

    model = RandomForestClassifier(n_estimators=N,max_depth=d,criterion='entropy')
    model.fit(X,Y)

    new_instance = np.asmatrix(testing[['mean_return','volatility']].values)
    prediction = model.predict(new_instance)
    return prediction

prediction = random_forest(traning, testing, 3, 5)
# trading by predict_labels
money= 100
# flag = 0 only have money 1 only have stock
flag = 0
portfolio = 100
portfolios, portfolios_buy_hold = [],[]
i = 0
for index, row in testing.iterrows():
    # buy hold
    if i == 0:
        portfolios_buy_hold.append(100)
        stock_buy_hold = 100/row['Adj Close']
    else:
        portfolios_buy_hold.append(stock_buy_hold*row['Adj Close'])

    # trading with labels_lda
    # red to green, buy stock
    if prediction[i] == 0 and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        portfolio = shares * row['Adj Close']
    # green to green, do nothing
    elif prediction[i] == 0 and flag == 1:
        portfolio = shares * row['Adj Close']
    # red to red, do nothing
    elif prediction[i] == 1 and flag == 0:
        pass
    # green to red, sell stock
    elif prediction[i] == 1 and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        portfolio = money
    portfolios.append(portfolio)

    i += 1

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
plt.legend(['trading with labels (random forest)', 'buy hold'])
plt.show()

if max([portfolios_buy_hold[-1],portfolios[-1]]) == portfolios_buy_hold[-1]:
    print('buy-and-hold results in a larger amount at the end of the year')
elif max([portfolios_buy_hold[-1], portfolios[-1]]) == portfolios[-1]:
    print('trading strategy based on random forest predicted labels results in a larger amount at the end of the year')
else:
    print('two strategies have equal amount at the end of the year')
