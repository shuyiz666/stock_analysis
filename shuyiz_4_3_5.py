'''
assignment3: KNN
question5: what is your true positive rate (sensitivity or recall) for 2018
'''
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')


df = pd.read_csv(ticker_file)
training = df[df['Year'] == 2017]
testing = df[df['Year'] == 2018]
X = training[['mean_return','volatility']].values
Y = training[['label']].values
testing_np = np.array(testing['label'])

k = 5
knn_classifier = KNeighborsClassifier(n_neighbors = k)
knn_classifier.fit(X,np.ravel(Y))
new_instance = testing[['mean_return','volatility']].values
prediction = knn_classifier.predict(new_instance)
cm = confusion_matrix(testing_np, prediction)
print('True positive = ', cm[0][0])
