'''
assignment3: KNN
question6: what is your true negative rate (specificity) for 2018?
'''
import os
import pandas as pd
from sklearn import svm
from sklearn . preprocessing import StandardScaler
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

svm_classifier = svm.SVC(kernel = 'rbf')
svm_classifier.fit(X,Y)
new_x = scaler.transform(testing[['mean_return','volatility']].values)
predicted = svm_classifier.predict(new_x)



# trading by predict_labels
money = 100
# flag = 0 only have money 1 only have stock
flag = 0
portfolio = 100
portfolios,portfolios_buy_hold = [],[]
i = 0
for index, row in testing.iterrows():
    # trading with labels
    # red to green, buy stock
    if predicted[i] == 'green' and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        portfolio = shares * row['Adj Close']
    # green to green, do nothing
    elif predicted[i] == 'green' and flag == 1:
        portfolio = shares * row['Adj Close']
    # red to red, do nothing
    elif predicted[i] == 'red' and flag == 0:
        pass
    # green to red, sell stock
    elif predicted[i] == 'red' and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        portfolio = money
    i += 1
    portfolios.append(portfolio)

print(portfolios[-1]-100)
