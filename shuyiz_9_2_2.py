'''
assignment2: decision trees
question2: compute the confusion matrix for year 2
'''

import os
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix
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

cm = confusion_matrix(testing['label'].values, prediction)
print('the confusion matrix for decision tree is:\n',cm)
