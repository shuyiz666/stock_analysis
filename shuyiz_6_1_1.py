'''
assignment1: Logistic regression
question1: what is the equation for logistic regression that your classifier found from year 1 data
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
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Y = training['label'].values
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X,Y)
print('the equation is:')
print('y =',round(log_reg_classifier.coef_[0][0],2),'* mean_return + ',round(log_reg_classifier.coef_[0][1],2),'* volatility +',round(log_reg_classifier.intercept_[0],2))
 
