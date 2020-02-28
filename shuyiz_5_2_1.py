'''
assignment2: custom kNN
question1: take three distance metrics: Euclidean (p = 2), Manhattan (p = 1) and generalized Minkovski for p = 1.5. For each value of p, compute the accuracy of your k-NN classifier on year 1 data. On x axis you plot p and on y-axis you plot accuracy. Which distance metric gives you the highest accuracy?
'''

import numpy as np
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

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
        training = self.X[['mean_return', 'volatility']].values
        testing = new_x[['mean_return', 'volatility']].values
        labels = []
        for i in testing:
            for j in range(len(training)):
                distance = np.linalg.norm(i - training[j], ord = self.distance_parameter_p)
                df_dists.loc[j] = [self.Labels[j], distance]
            Sorted_df_dists = df_dists.sort_values(by='distance', ascending=True)
            toplabel = Sorted_df_dists['label'][0:5]
            # freq label, frequency
            predict_label = Counter(toplabel).most_common(1)[0][0]
            labels.append(predict_label)
        self.labels = labels
        return self.labels

    def draw_decision_boundary(self, new_x):
        x = new_x['mean_return'].values
        y = new_x['volatility'].values

        id_list = new_x['Week_Number'].values

        plt.xlabel('mean')
        plt.ylabel('volatility')
        plt.plot([1, 2.2], [0, 18], color='black', ls='dotted')
        plt.scatter(x, y, color=np.array(self.labels))
        for i, txt in enumerate(id_list):
            plt.text(x[i] + 0.2, y[i] + 0.2, txt, fontsize=5)
        plt.show()

if __name__ == '__main__':
    wd = os.getcwd()
    ticker = 'ZSAN'
    input_dir = wd
    ticker_file = os.path.join(input_dir, ticker + '_label.csv')
    df = pd.read_csv(ticker_file)

    training = df[df['Year'] == 2017]
    Labels = training['label'].values

    p1 = Custom_knn(5,1)
    p1.fit(training, Labels)
    predict1 = p1.predict(training)
    accuracy1 = round(sum(predict1 == Labels) / len(predict1), 2)
    print('accuracy for Manhattan distance is:', accuracy1)

    p1_5 = Custom_knn(5,1.5)
    p1_5.fit(training, Labels)
    predict1_5 = p1_5.predict(training)
    accuracy1_5 = round(sum(predict1_5 == Labels) / len(predict1_5), 2)
    print('accuracy for Minkovski distance for p = 1.5 is:', accuracy1_5)

    p2 = Custom_knn(5,2)
    p2.fit(training, Labels)
    predict2 = p2.predict(training)
    accuracy2 = round(sum(predict2 == Labels) / len(predict2), 2)
    print('accuracy for Euclidean distance is:', accuracy2)


    accuracys = [accuracy1, accuracy1_5, accuracy2]
    ps = [1,1.5,2]
    plt.title('k-NN classifier p and accuracy in year1')
    plt.xlabel('p', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.plot(ps, accuracys)
    plt.show()

    print('accuracy for all distance are same which are all 88% ')







