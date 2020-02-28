'''
assignment2: custom kNN
question3: take p = 1.5. In year 2, pick two weeks for which your classi-
fier gave different labels. Use method display decision boundary() to show the neighbors (both colors and ids)
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
        x = new_x['mean_return'].values
        y = new_x['volatility'].values

        id_list = new_x['Week_Number'].values

        plt.xlabel('mean')
        plt.ylabel('volatility')
        plt.plot([1, 11 / 3], [0, 40], color='black', ls='dotted')
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

    testing = df[df['Year'] == 2018]


    p1_5 = Custom_knn(5, 1.5)
    p1_5.fit(training, Labels)
    p1_5.predict(testing)
    p1_5.draw_decision_boundary(testing)









