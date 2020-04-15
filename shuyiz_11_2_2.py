'''
assignment1: my own k-means
question1:do you get different cluster?
What distance metric gives you the most “pure” clusters?
'''
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
testing = df[df['Year']==2018]

n_clusters = 2
initial_centroids0 = testing[['mean_return','volatility']].values[0]
initial_centroids1 = testing[['mean_return','volatility']].values[1]

def kmeans(testing,initial_centroids0,initial_centroids1,dis):
    X = testing[['mean_return', 'volatility']].values
    cluster0,cluster1,cluster = [],[],[]
    SSE = 0
    for i in range(len(X)):
        distance0 = np.linalg.norm(initial_centroids0 - X[i], ord=dis)
        distance1 = np.linalg.norm(initial_centroids1 - X[i], ord=dis)
        if distance0 >= distance1:
            SSE += distance1**2
            cluster1.append(X[i])
            cluster.append(1)
        else:
            SSE += distance0 ** 2
            cluster0.append(X[i])
            cluster.append(0)
    centroids0 = np.mean(cluster0,axis=0)
    centroids1 = np.mean(cluster1,axis=0)
    if (centroids0 == initial_centroids0).all() and (centroids1 == initial_centroids1).all():
        return cluster
    else:
        return kmeans(testing,centroids0,centroids1,dis)

def percentage_label(testing,cluster):
    testing['cluster'] = cluster
    for i in range(2):
        new_df = testing[testing['cluster'] == i]
        print('cluser', i, ',', 'green label:',
              '%s%%' % (round(sum(new_df['label'] == 'green') / len(new_df) * 100, 2)), 'red label:',
              '%s%%' % (round(sum(new_df['label'] == 'red') / len(new_df) * 100, 2)))


Euclidean = kmeans(testing,initial_centroids0,initial_centroids1,2)
Manhattan = kmeans(testing,initial_centroids0,initial_centroids1,1)
Minkowski = kmeans(testing,initial_centroids0,initial_centroids1,1.5)

print('Euclidean:')
Euclidean_result = percentage_label(testing,Euclidean)
print('Manhattan:')
Manhattan_result = percentage_label(testing,Manhattan)
print('Minkowski:')
Minkowski_result = percentage_label(testing,Minkowski)
print('I get different clusters, there is no pure clusters in all distance models')