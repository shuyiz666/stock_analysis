'''
assignment2: custom kNN
question4: compute the confusion matrices for p = 1, p = 1.5 and p=2
'''

import numpy as np
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class Custom_knn():
    def __init__(self, number_neighbots_k, distance_parameter_p):
        self.number_neighbots_k = number_neighbots_k
        self.distance_parameter_p = distance_parameter_p

    def __str__(self):
        return "number_neightbors_k = " + str(self.number_neighbors_k) + ", p = " + str(self.distance_parameter_p)

    def fit(self, X, Labels):
        # training
        self.X = X
        # class
        self.Labels = Labels

    def predict(self, new_x):
        df_dists = pd.DataFrame(columns=['Week_Number','label', 'distance'])
        training = self.X[['mean_return', 'volatility']].values
        testing = new_x[['mean_return', 'volatility']].values
        labels,neighbors = [],[]
        for i in testing:
            for j in range(len(training)):
                distance = np.linalg.norm(i - training[j], ord = self.distance_parameter_p)
                df_dists.loc[j] = [self.X['Week_Number'][j], self.Labels[j], distance]
            Sorted_df_dists = df_dists.sort_values(by='distance', ascending=True)
            toplabel = Sorted_df_dists['label'][0:5]
            # 5 neighbor for each row of new_x
            neighbor = Sorted_df_dists['Week_Number'][0:5].values

            # freq label, frequency
            predict_label = Counter(toplabel).most_common(1)[0][0]
            labels.append(predict_label)
            neighbors.append(neighbor)
        self.labels = labels
        # neighbors list
        self.neighbors = neighbors
        return self.labels, self.neighbors

    def draw_decision_boundary(self, new_x):
        x = self.X['mean_return'].values
        y = self.X['volatility'].values
        id_list = self.X['Week_Number'].values

        plt.title('knn predict labels with neighbors')
        plt.xlabel('mean')
        plt.ylabel('volatility')
        plt.plot([1, 2.2], [0, 18], color='black', ls='dotted')
        plt.scatter(x, y, color=np.array(self.Labels))
        for i, txt in enumerate(id_list):
            plt.text(x[i] + 0.2, y[i] + 0.2, txt, fontsize=5)

        x_new = new_x['mean_return'].values
        y_new = new_x['volatility'].values
        plt.scatter(x_new, y_new, color = self.labels,marker="*",s=80)

        row = 0
        while row < len(x_new):
            # neighbor for each row of new_x
            for neighbor_week_number in self.neighbors[row]:
                neighbor_x = self.X.loc[self.X['Week_Number']==neighbor_week_number,'mean_return'].values[0]
                neighbor_y = self.X.loc[self.X['Week_Number']==neighbor_week_number,'volatility'].values[0]
                plt.plot([neighbor_x,x_new[row]],[neighbor_y,y_new[row]], color='darkgrey', ls='dotted')
            row += 1

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
    predict1, neighbors1 = p1.predict(testing)
    cm1 = confusion_matrix(testing['label'].values, predict1)
    print('confusion matrix for p = 1 is:\n', cm1, '\n')

    p1_5 = Custom_knn(5, 1.5)
    p1_5.fit(training, Labels)
    predict1_5, neighbors1_5 = p1_5.predict(testing)
    cm1_5 = confusion_matrix(testing['label'].values, predict1_5)
    print('confusion matrix for p = 1.5 is:\n', cm1_5, '\n')

    p2 = Custom_knn(5, 2)
    p2.fit(training, Labels)
    predict2, neighbors2 = p2.predict(testing)
    cm2 = confusion_matrix(testing['label'].values, predict2)
    print('confusion matrix for p = 2 is:\n', cm2, '\n')








