'''
assignment2: custom kNN
question6: for p = 1, p = 1.5 and p = 2 implement a trading strategy based on your labels for year 2 and compare the perfor- mance with the ”buy-and-hold” strategy. For which value of p does your strategy result in the largest portfolio value at the end of the year?
'''

import numpy as np
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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

def buy_hold(df):
    portfolios = []
    x = 0
    for index, row in df.iterrows():
        # buy hold strategy
        if x == 0:
            portfolios.append(100)
            x += 1
            stock = 100 / row['Adj Close']
        else:
            portfolios.append(stock * row['Adj Close'])
    return portfolios


def trade_with_label(df):
    money = 100
    # flag = 0 only have money 1 only have stock
    flag = 0
    portfolio = 100
    portfolios = []
    for index, row in df.iterrows():
        # red to green, buy stock
        if row['predict_label'] == 'green' and flag == 0:
            shares = money / row['Adj Close']
            money = 0
            flag = 1
            portfolio = shares * row['Adj Close']
        # green to green, do nothing
        elif row['predict_label'] == 'green' and flag == 1:
            portfolio = shares * row['Adj Close']
        # red to red, do nothing
        elif row['predict_label'] == 'red' and flag == 0:
            pass
        # green to red, sell stock
        elif row['predict_label'] == 'red' and flag == 1:
            money = shares * row['Adj Close']
            shares = 0
            flag = 0
            portfolio = money
        portfolios.append(portfolio)
    return portfolios

if __name__ == '__main__':
    wd = os.getcwd()
    ticker = 'ZSAN'
    input_dir = wd
    ticker_file = os.path.join(input_dir, ticker + '_label.csv')
    df = pd.read_csv(ticker_file)

    training = df[df['Year'] == 2017]
    Labels = training['label'].values

    testing = df[df['Year'] == 2018]
    portfolios_buy_hold = buy_hold(testing)
    print('the portfolio value at the end of the year for buy hold is',round(portfolios_buy_hold[-1],2))

    p1 = Custom_knn(5, 1)
    p1.fit(training, Labels)
    predict1, neighbors1 = p1.predict(testing)
    testing_predict1 = testing.copy()
    testing_predict1['predict_label'] = predict1
    portfolios_p1 = trade_with_label(testing_predict1)
    print('the portfolio value at the end of the year for p=1 is',round(portfolios_p1[-1],2))

    p1_5 = Custom_knn(5, 1.5)
    p1_5.fit(training, Labels)
    predict1_5, neighbors1_5 = p1_5.predict(testing)
    testing_predict1_5 = testing.copy()
    testing_predict1_5['predict_label'] = predict1_5
    portfolios_p1_5 = trade_with_label(testing_predict1_5)
    print('the portfolio value at the end of the year for p=1.5 is', round(portfolios_p1_5[-1],2))

    p2 = Custom_knn(5, 2)
    p2.fit(training, Labels)
    predict2, neighbors2 = p2.predict(testing)
    testing_predict2 = testing.copy()
    testing_predict2['predict_label'] = predict2
    portfolios_p2 = trade_with_label(testing_predict2)
    print('the portfolio value at the end of the year for p=2 is', round(portfolios_p2[-1],2))


    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    plt.title('portfolio in different strategies')
    plt.xlabel('week numbers')
    plt.ylabel('portfolio')
    plt.yticks()
    plt.plot(testing['Week_Number'], portfolios_buy_hold)
    plt.plot(testing['Week_Number'], portfolios_p1)
    plt.plot(testing['Week_Number'], portfolios_p1_5)
    plt.plot(testing['Week_Number'], portfolios_p2)
    plt.legend(['buy hold','p=1','p=1.5','p=2'])
    plt.show()

    print('p = 1 results the largest portfolio value at the end of the year ')







 

