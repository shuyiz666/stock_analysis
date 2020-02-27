'''
assignment2: custom kNN
question1: take three distance metrics: Euclidean (p = 2), Manhattan (p = 1) and generalized Minkovski for p = 1.5. For each value of p, compute the accuracy of your k-NN classifier on year 1 data. On x axis you plot p and on y-axis you plot accuracy. Which distance metric gives you the highest accuracy?
'''

import numpy as np
import os
import pandas as pd
from collections import Counter

class Custom_knn():
    def __init__(self,number_neighbots_k, distance_parameter_p):
        self.number_neighbots_k = number_neighbots_k
        self.distance_parameter_p = distance_parameter_p

    def __str__(self):
        pass

    def fit(self, X, Labels):
        # X 训练集 只有mean和sigma
        self.X = X
        # Labels 训练集 只有label
        self.Labels = Labels

    def predict(self, new_x):
        df_dists = pd.DataFrame(columns=['label', 'distance'])
        labels = []
        for i in new_x:
            for j in range(len(X)):
                distance = np.linalg.norm(i - X[j], ord = self.distance_parameter_p)
                df_dists.loc[j] = [self.Labels[j], distance]
            Sorted_df_dists = df_dists.sort_values(by='distance', ascending=True)
            toplabel = Sorted_df_dists['label'][0:5]
            # freq label, frequency
            predict_label = Counter(toplabel).most_common(1)[0][0]
            labels.append(predict_label)
        return labels

    def draw_decision_boundary(self, new_x):

if __name__ == '__main__':
    wd = os.getcwd()
    ticker = 'ZSAN'
    input_dir = wd
    ticker_file = os.path.join(input_dir, ticker + '_label.csv')
    df = pd.read_csv(ticker_file)

    training = df[df['Year'] == 2017]
    X = training[['mean_return', 'volatility']].values
    testing = df[df['Year'] == 2018]
    new_x = testing[['mean_return', 'volatility']].values






