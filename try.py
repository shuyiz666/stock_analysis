'''
assignment3: KNN
question1: take k = 3,5,7,9,11. For each value of k compute the accuracy of your k-NN classifier on 2017 data. On x axis you plot k and on y-axis you plot accuracy.
'''
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

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
accuracy_list = []
ks = [3]

for k in ks:
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X,np.ravel(Y))
    new_instance = testing[['mean_return','volatility']].values
    prediction = knn_classifier.predict(new_instance)
    print(prediction[1])
    accuracy = round(sum(testing_np==prediction)/len(testing_np),4)
    accuracy_list.append(accuracy)

#
# plt.title('k and accuracy')
# plt.xlabel('k')
# plt.ylabel('accuracy')
# plt.plot(ks, accuracy_list)
# plt.show()