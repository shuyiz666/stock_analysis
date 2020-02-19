'''
assignment3: KNN
question3: use the optimal value of k from 2017 to predict labels for
2018. What is your accuracy
'''
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

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
accuracy = round(sum(testing_np==prediction)/len(testing_np),2)
print('the accuracy is', accuracy)