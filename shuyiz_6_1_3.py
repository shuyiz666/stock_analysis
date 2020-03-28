'''
assignment1: Logistic regression
question3: compute the confusion matrix for year 2
'''
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

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
cm = confusion_matrix(testing['label'], predicted)
print('confusion matrix is:\n',cm,'\n') 
