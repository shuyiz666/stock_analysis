'''
assignment1: k-means
question1:  take k = 3 and use k-means sklearn library routing for kmeans (random initialization and use the defaults).
Take k = 1, 2, . . . 7, 8 and compute the distortion vs. k. Use the
”knee” method to find out the best k.
'''
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
traning = df[df['Year']==2017]
n_clusters = 3

# a. k-means for k = 3
X = traning[['mean_return','volatility']].values
kmeans_classifier = KMeans(n_clusters = n_clusters)
y_means = kmeans_classifier.fit_predict(X)
traning['cluster'] = y_means
print('when k = 3, the predicted labels are:',y_means)
colmap = {0: 'blue', 1:'red', 2:'yellow'}
fig = plt.figure()
for i in range(n_clusters):
    new_df = traning[traning['cluster']==i]
    plt.scatter(new_df['mean_return'],new_df['volatility'],color = colmap[i], s=20, label='points in cluster' + str(i+1))
for i in range(n_clusters):
    plt.scatter(kmeans_classifier.cluster_centers_[i][0], kmeans_classifier.cluster_centers_[i][1], color=colmap[i],
                marker='x', s = 50, label ='centroid' + str(i + 1))
for i in range(len(X)):
    x_text = traning['mean_return'].iloc[i] + 0.05
    y_text = traning['volatility'].iloc[i] + 0.2
    id_text = traning['Week_Number'].iloc[i]
    plt.text(x_text, y_text, s = id_text,fontsize=10)
plt.title('k-means result when k = 3')
plt.legend(loc='upper left')
plt.xlabel('mean_return')
plt.ylabel('volatility')
plt.show()

# b. plot different k
inertia_list = []
for k in range (1 ,9):
    kmeans_classifier = KMeans(n_clusters=k)
    y_kmeans = kmeans_classifier.fit_predict(X)
    inertia = kmeans_classifier.inertia_ # SSE
    inertia_list.append(inertia)
fig, ax = plt.subplots(1,figsize=(7,5))
plt.plot(range(1, 9), inertia_list , marker='o',color='green')
plt.xlabel('number of clusters: k')
plt.ylabel('inertia')
plt.tight_layout()
plt.show()
print('the best k is 2')