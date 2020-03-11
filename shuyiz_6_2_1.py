'''
assignment2: Weekly Trading with Linear Models
question1: take weekly data for year 1. For each W = 5,6,...,12 and for each d = 1, 2, 3 construct the corresponding polynomials Use these polynomials to predict weekly labels. Plot the accuracy - on x axis you have W and you plot three curves for accuracy (separate curve for each d)
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def linear_model(W,d,training):
    x = training['Week_Number'].values
    y = training['Adj Close'].values
    start = 0
    end = W
    results, real_results = [], []
    while end < len(x):
        train_x = x[start:end] # week number
        train_y = y[start:end] # close price
        testing_x = x[end]
        weights = np.polyfit(train_x,train_y,d)
        model = np.poly1d(weights)
        predicted = model(testing_x)
        real_label = training[training['Week_Number']==testing_x]['label'].values[0]
        real_results.append(real_label)
        if predicted > y[end-1]:
            results.append('green')
        elif predicted < y[end-1]:
            results.append('red')
        else:
            results.append(training[training['Week_Number'] == end - 1]['label'].values[0])
        start += 1
        end += 1
    return results,real_results

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
training = df[df['Year']==2017]

plt.title('accuracy change with d and W')
plt.xlabel('W')
plt.ylabel('accuracy')

W_range = list(range(5,13))
legend = []
for d in range(1,4):
    y_axis = []
    for W in W_range:
        results, real_results = linear_model(W,d,training)
        accuracy = '%s%%'%round(sum(np.array(results)==np.array(real_results))/len(results)*100,0)
        y_axis.append(accuracy)
    plt.plot(W_range, y_axis)
    legend.append('d='+str(d))

plt.legend(legend)
plt.show()