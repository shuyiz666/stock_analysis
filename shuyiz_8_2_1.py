'''
assignment2: naive bayesian
question1: implement a Gaussian naive bayesian classifier and compute its accuracy for year 2
'''
import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
traning = df[df['Year']==2017]
testing = df[df['Year']==2018]

X = traning[['mean_return','mean_return']].values
Y = traning[['label']].values

NB_classifier = GaussianNB().fit(X,Y.ravel())

new_instance = np.asmatrix(testing[['mean_return','mean_return']].values)
prediction = NB_classifier.predict(new_instance)
print('accuracy for year 2 is','%s%%'%round(sum(prediction==testing[['label']].values.ravel())/len(testing[['label']].values.ravel())*100,2))
