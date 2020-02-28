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

    new_x = testing[testing.Week_Number.isin([5,26])]

    p1_5 = Custom_knn(5, 1.5)
    p1_5.fit(training, Labels)
    labels, neighbors = p2.predict(new_x)
    p1_5.draw_decision_boundary(new_x)








