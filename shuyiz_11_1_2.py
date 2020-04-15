'''
assignment1: k-means
question2:  for this optimal k, examine your clusters and for each cluster compute the percentage of ”green” and ”red” weeks in that cluster.
'''
import os
import pandas as pd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
testing = df[df['Year']==2018]
n_clusters = 2

# a. k-means for k = 3
X = testing[['mean_return','volatility']].values
kmeans_classifier = KMeans(n_clusters = n_clusters)
y_means = kmeans_classifier.fit_predict(X)
testing['cluster'] = y_means

for i in range(n_clusters):
    new_df = testing[testing['cluster']==i]
    print('cluser',i,',','green label:','%s%%' % (round(sum(new_df['label'] == 'green')/len(new_df)* 100, 2)),'red label:','%s%%' % (round(sum(new_df['label'] == 'red')/len(new_df)* 100, 2)))