'''
assignment2: decision trees
question1: implement a decision tree and compute its accuracy for year 2
'''

import os
import pandas as pd
import numpy as np
from sklearn import tree
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
traning = df[df['Year']==2017]
testing = df[df['Year']==2018]

X = traning[['mean_return', 'volatility']].values
Y = traning[['label']].values

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X,Y)

new_instance = np.asmatrix(testing[['mean_return','volatility']].values)
prediction = clf.predict(new_instance)

accuracy = sum(testing['label'].values==prediction)/len(prediction)
print('the accuracy for year2 for decision tree is','%s%%'%round(accuracy*100,2))