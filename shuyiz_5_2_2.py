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
        # training
        self.X = X
        # class
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
        x = self.X['mean_return'].values
        y = self.X['volatility'].values
        id_list = self.X['Week_Number'].values

        plt.xlabel('mean')
        plt.ylabel('volatility')
        plt.plot([1, 2.2], [0, 18], color='black', ls='dotted')
        plt.scatter(x, y, color=np.array(self.Labels))
        for i, txt in enumerate(id_list):
            plt.text(x[i] + 0.2, y[i] + 0.2, txt, fontsize=5)

        # x_new = new_x['mean_return'].values
        # y_new = new_x['volatility'].values
        # plt.scatter(x_new, y_new, color = 'blue')
        plt.show()


if __name__ == '__main__':
    wd = os.getcwd()
    ticker = 'ZSAN'
    input_dir = wd
    ticker_file = os.path.join(input_dir, ticker + '_label.csv')
    df = pd.read_csv(ticker_file)

    training = df[df['Year'] == 2017]
    Labels = training['label'].values

    testing = df[df['Year'] == 2018]

    p1 = Custom_knn(5, 1)
    p1.fit(training, Labels)
    predict1 = p1.predict(testing)
    accuracy1 = round(sum(predict1 == testing['label']) / len(predict1), 2)
    print('accuracy for Manhattan distance is:', accuracy1)

    p1_5 = Custom_knn(5, 1.5)
    p1_5.fit(training, Labels)
    predict1_5 = p1_5.predict(testing)
    accuracy1_5 = round(sum(predict1_5 == testing['label']) / len(predict1_5), 2)
    print('accuracy for Minkovski distance for p = 1.5 is:', accuracy1_5)

    p2 = Custom_knn(5, 2)
    p2.fit(training, Labels)
    predict2 = p2.predict(testing)
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







