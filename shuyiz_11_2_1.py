'''
assignment1: my own k-means
question1:  You take k* that you computed before and apply and compare these three choices (for the same initial choice of centroids)
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
testing = df[df['Year']==2018]
X = testing[['mean_return','volatility']].values
n_clusters = 2
initial_centroids0 = X[0]
initial_centroids1 = X[1]

def kmeans(X,initial_centroids0,initial_centroids1,dis):
    cluster0,cluster1 = [],[]
    SSE = 0
    for i in range(len(X)):
        distance0 = np.linalg.norm(initial_centroids0 - X[i], ord=dis)
        distance1 = np.linalg.norm(initial_centroids1 - X[i], ord=dis)
        if distance0 >= distance1:
            SSE += distance1**2
            cluster1.append(X[i])
        else:
            SSE += distance0 ** 2
            cluster0.append(X[i])
    centroids0 = np.mean(cluster0,axis=0)
    centroids1 = np.mean(cluster1,axis=0)
    if (centroids0 == initial_centroids0).all() and (centroids1 == initial_centroids1).all():
        return SSE
    else:
        return kmeans(X,centroids0,centroids1,dis)

SSE_Euclidean = kmeans(X,initial_centroids0,initial_centroids1,2)
SSE_Manhattan = kmeans(X,initial_centroids0,initial_centroids1,1)
SSE_Minkowski = kmeans(X,initial_centroids0,initial_centroids1,1.5)

print('SSE for Euclidean is',round(SSE_Euclidean,2))
print('SSE for Manhattan is',round(SSE_Manhattan,2))
print('SSE for SSE_Minkowski(p=1.5) is',round(SSE_Minkowski,2))
print('Euclidean is the best')