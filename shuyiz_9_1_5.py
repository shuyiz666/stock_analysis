'''
assignment1: Discriminant Analysis
question5: implement trading strategyies based on your labels for year 2 (for both linear and quadratic) and compare the perfor- mance with the ”buy-and-hold” strategy. Which strategy results in a larger amount at the end of the year?
'''
import os
import pandas as pd
from sklearn . discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn . discriminant_analysis import QuadraticDiscriminantAnalysis as LQA
from sklearn . preprocessing import StandardScaler
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

X = traning[['mean_return','volatility']].values
scaler = StandardScaler()
scaler .fit(X)

X = scaler.transform(X)
Y = traning[['label']].values.ravel()

def lda(X,Y):
    lda_classifier = LDA(n_components = 2)
    lda_classifier.fit(X,Y)
    new_instance = scaler.transform(testing[['mean_return', 'volatility']].values)
    predicted = lda_classifier.predict(new_instance)
    return predicted

def qda(X,Y):
    qda_classifier = LQA()
    qda_classifier.fit(X,Y)
    new_instance = scaler.transform(testing[['mean_return', 'volatility']].values)
    predicted = qda_classifier.predict(new_instance)
    return predicted

lda = lda(X,Y)
qda = qda(X,Y)

# trading by predict_labels
money_lda,money_qda = 100,100
# flag = 0 only have money 1 only have stock
flag_lda,flag_qda = 0,0
portfolio_lda, portfolio_qda= 100, 100
portfolios_lda, portfolios_qda, portfolios_buy_hold = [],[],[]
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
    if lda[i] == 'green' and flag_lda == 0:
        shares_lda = money_lda / row['Adj Close']
        money_lda = 0
        flag_lda = 1
        portfolio_lda = shares_lda * row['Adj Close']
    # green to green, do nothing
    elif lda[i] == 'green' and flag_lda == 1:
        portfolio_lda = shares_lda * row['Adj Close']
    # red to red, do nothing
    elif lda[i] == 'red' and flag_lda == 0:
        pass
    # green to red, sell stock
    elif lda[i] == 'red' and flag_lda == 1:
        money_lda = shares_lda * row['Adj Close']
        shares_lda = 0
        flag_lda = 0
        portfolio_lda = money_lda
    portfolios_lda.append(portfolio_lda)

    # trading with labels_qda
    # red to green, buy stock
    if qda[i] == 'green' and flag_qda == 0:
        shares_qda = money_qda / row['Adj Close']
        money_qda = 0
        flag_qda = 1
        portfolio_qda = shares_qda * row['Adj Close']
    # green to green, do nothing
    elif qda[i] == 'green' and flag_qda == 1:
        portfolio_qda = shares_qda * row['Adj Close']
    # red to red, do nothing
    elif qda[i] == 'red' and flag_qda == 0:
        pass
    # green to red, sell stock
    elif qda[i] == 'red' and flag_qda == 1:
        money_qda = shares_qda * row['Adj Close']
        shares_qda = 0
        flag_qda = 0
        portfolio_qda = money_qda
    portfolios_qda.append(portfolio_qda)

    i += 1

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.title('three strategies portfolio in 2018')
plt.xlabel('week_number', fontsize=14)
xlabels = testing['Week_Number']
plt.ylabel('portfolio', fontsize=14)
plt.yticks()
plt.plot(xlabels,portfolios_lda)
plt.plot(xlabels,portfolios_qda)
plt.plot(xlabels,portfolios_buy_hold)
plt.legend(['trading with labels (linear classifier)','trading with labels (quadratic classifier)', 'buy hold'])
plt.show()

if max([portfolios_buy_hold[-1],portfolios_lda[-1],portfolios_qda[-1]]) == portfolios_buy_hold[-1]:
    print('buy-and-hold results in a larger amount at the end of the year')
elif max([portfolios_buy_hold[-1],portfolios_lda[-1],portfolios_qda[-1]]) == portfolios_lda[-1]:
    print('trading strategy based on linear classifier predicted labels results in a larger amount at the end of the year')
elif max([portfolios_buy_hold[-1],portfolios_lda[-1],portfolios_qda[-1]]) == portfolios_qda[-1]:
    print('trading strategy based on quadratic classifier predicted labels results in a larger amount at the end of the year')
