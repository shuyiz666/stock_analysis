'''
assignment3: KNN
question1: take k = 3,5,7,9,11. For each value of k compute the accuracy of your k-NN classifier on 2017 data. On x axis you plot k and on y-axis you plot accuracy.
'''
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')


df = pd.read_csv(ticker_file)
df2017 = df[df['Year'] == 2017]
X = df2017[['mean_return','volatility']].values
Y = df2017[['label']].values
content_train, content_test, label_train, label_test = train_test_split(X, Y, test_size=0.33,
                                                                        random_state=0)

# testing_np = np.array(testing['label'])
accuracy_list = []
ks = [3,5,7,9,11]

for k in ks:
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(content_train,np.ravel(label_train))
    new_instance = content_test
    prediction = knn_classifier.predict(new_instance)
    accuracy = round(metrics.accuracy_score(label_test, prediction),4)
    accuracy_list.append(accuracy)

plt.title('k and accuracy')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(ks, accuracy_list)
plt.show()
