'''
assignment1: kNN (using sklearn) add results of trading using knn (we only trade year 2)
'''
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as mtick

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

df = pd.read_csv(ticker_file)
training = df[df['Year'] == 2017].copy()
testing = df[df['Year'] == 2018].copy()
X = training[['mean_return','volatility']].values
Y = training[['label']].values
testing_np = np.array(testing['label'])

# find optimal value of k
accuracy_dic = {}
ks = [3,5,7,9,11]
for k in ks:
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X,np.ravel(Y))
    new_instance = testing[['mean_return','volatility']].values
    prediction = knn_classifier.predict(new_instance)
    accuracy = round(sum(testing_np==prediction)/len(testing_np),4)
    accuracy_dic[k] = accuracy

plt.title('k and accuracy')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(ks, list(accuracy_dic.values()))
plt.show()

bestk = max(accuracy_dic, key=accuracy_dic.get)
print('the optimal value of k is:', bestk)

# use the optimal value of k from 2017 to predict labels for 2018
knn_classifier = KNeighborsClassifier(n_neighbors = bestk)
knn_classifier.fit(X,np.ravel(Y))
new_instance = testing[['mean_return','volatility']].values
prediction = knn_classifier.predict(new_instance)
accuracy = sum(testing_np==prediction)/len(testing_np)
print('the accuracy is', '%s%%'%(round(accuracy*100,2)))
testing['predict_label'] = prediction

# compute the confusion matrix for 2018
cm = confusion_matrix(testing['label'], testing['predict_label'])
print('confusion matrix is:\n',cm,'\n')

# true positive rate and true negative rate for 2018
print('True positive = ', cm[0][0])
print('True negative = ', cm[1][1])

# trading result with predict labels
money = 100
# flag = 0 only have money 1 only have stock
flag = 0
value = 100
values_hold = []
x = 0
for index, row in testing.iterrows():
    # red to green, buy stock
    if row['predict_label'] == 'green' and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        value = shares * row['Adj Close']
    # green to green, do nothing
    elif row['predict_label'] == 'green' and flag == 1:
        value = shares * row['Adj Close']
    # red to red, do nothing
    elif row['predict_label'] == 'red' and flag == 0:
        value = value
    # green to red, sell stock
    elif row['predict_label'] == 'red' and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        value = money
    values_hold.append(value)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.title('trading with predict labels in 2018')
plt.xlabel('week_number', fontsize=14)
xlabels = testing['Week_Number']
plt.ylabel('portfolio', fontsize=14)
plt.yticks()
plt.plot(xlabels,values_hold)
plt.show()