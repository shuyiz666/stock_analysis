'''
assignment1: Logistic regression
question4: what is true positive rate (sensitivity or recall) and true negative rate (specificity) for year 2
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
true_positive_rate = cm[0][0]/(cm[0][0]+cm[0][1])
true_negative_rate = cm[1][1]/(cm[1][1]+cm[1][0])
print('true positive rate =','%s%%'%round(true_positive_rate*100,2))
print('true negative rate =','%s%%'%round(true_negative_rate*100,2)) 
