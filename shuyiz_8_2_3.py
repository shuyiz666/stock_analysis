'''
assignment2: naive bayesian
question3: what is true positive rate and true negative rate for year 2
'''
import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
traning = df[df['Year']==2017]
testing = df[df['Year']==2018]

X = traning[['mean_return','volatility']].values
Y = traning[['label']].values

NB_classifier = GaussianNB().fit(X,Y.ravel())

new_instance = np.asmatrix(testing[['mean_return','volatility']].values)
prediction = NB_classifier.predict(new_instance)

cm = confusion_matrix(testing[['label']].values.ravel(), prediction)
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
TPR = TP/(TP+FN)
TNR = TN/(TN+FP)

print('true positive rate is','%s%%'%round(TPR*100,2))
print('true negative rate is','%s%%'%round(TNR*100,2)) 
