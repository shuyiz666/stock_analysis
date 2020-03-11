'''
assignment1: Logistic regression
question2: what is the accuracy for year 2
'''
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)

training = df[df['Year']==2017]
testing = df[df['Year']==2018]
X = training[['mean_return', 'volatility']].values
new = testing[['mean_return', 'volatility']].values
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Y = training['label'].values
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X,Y)
new_x = scaler.transform(new)
predicted = log_reg_classifier.predict(new_x)
accuracy = sum(predicted == testing['label'])/len(predicted)
print('accuracy in year2:\n','%s%%'%round(accuracy*100,2))