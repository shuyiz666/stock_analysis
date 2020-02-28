'''
assignment2: custom kNN
question2: repeat this for year 2 and plot your results. Are there any differences with year 1
'''

import numpy as np
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


class Custom_knn():
    def __init__(self, number_neighbots_k, distance_parameter_p):
        self.number_neighbots_k = number_neighbots_k
        self.distance_parameter_p = distance_parameter_p

    def __str__(self):
        pass

    def fit(self, X, Labels):
        # attributes
        self.X = X
        # class
        self.Labels = Labels

    def predict(self, new_x):
        df_dists = pd.DataFrame(columns=['label', 'distance'])
        labels = []
        for i in new_x:
            for j in range(len(X)):
                distance = np.linalg.norm(i - X[j], ord=self.distance_parameter_p)
                df_dists.loc[j] = [self.Labels[j], distance]
            Sorted_df_dists = df_dists.sort_values(by='distance', ascending=True)
            toplabel = Sorted_df_dists['label'][0:5]
            # freq label, frequency
            predict_label = Counter(toplabel).most_common(1)[0][0]
            labels.append(predict_label)
        self.labels = labels
        return self.labels

    def draw_decision_boundary(self, new_x):
        rs = []
        sigmas = []
        for [r, sigma] in new_x:
            rs.append(r)
            sigmas.append(sigma)
        plt.scatter(rs, sigmas, color=np.array(self.labels), s=20, alpha=0.5)
        plt.title('KNN prediction plot')
        plt.xlabel('mean')
        plt.ylabel('volatility')
        plt.show()


if __name__ == '__main__':
    wd = os.getcwd()
    ticker = 'ZSAN'
    input_dir = wd
    ticker_file = os.path.join(input_dir, ticker + '_label.csv')
    df = pd.read_csv(ticker_file)

    training = df[df['Year'] == 2017]
    X = training[['mean_return', 'volatility']].values
    Labels = training['label'].values

    testing = df[df['Year'] == 2018]
    new_x = testing[['mean_return', 'volatility']].values

    p1 = Custom_knn(5, 1)
    p1.fit(X, Labels)
    predict1 = p1.predict(new_x)
    accuracy1 = round(sum(predict1 == testing['label']) / len(predict1), 2)
    print('accuracy for Manhattan distance is:', accuracy1)

    p1_5 = Custom_knn(5, 1.5)
    p1_5.fit(X, Labels)
    predict1_5 = p1_5.predict(new_x)
    accuracy1_5 = round(sum(predict1_5 == testing['label']) / len(predict1_5), 2)
    print('accuracy for Minkovski distance for p = 1.5 is:', accuracy1_5)

    p2 = Custom_knn(5, 2)
    p2.fit(X, Labels)
    predict2 = p2.predict(new_x)
    accuracy2 = round(sum(predict2 == testing['label']) / len(predict2), 2)
    print('accuracy for Euclidean distance is:', accuracy2)

    accuracys = [accuracy1, accuracy1_5, accuracy2]
    ps = [1, 1.5, 2]
    plt.title('k-NN classifier p and accuracy in year2')
    plt.xlabel('p', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.plot(ps, accuracys)
    plt.show()

    print('Minkovski and Euclidean both have highest accuracy which is 87%')







