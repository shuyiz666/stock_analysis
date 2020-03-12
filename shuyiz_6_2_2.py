'''
assignment2: Weekly Trading with Linear Models
question2:  for each d take the best W that gives you the highest accu- racy. Use this W to predict labels for year 2. What is your accuracy?
'''

import os
import pandas as pd
import numpy as np

def linear_model(W,d,df):
    index_year2 = df[df['Year'] == 2018].index.values.astype(int)[0]
    x = df['Week_Number'].values
    y = df['Adj Close'].values
    start = index_year2-W
    end = index_year2
    results, real_results = [], []
    while end <= df.index[-1]:
        train_x = x[start:end] # week number
        train_y = y[start:end] # close price
        testing_x = x[end]
        weights = np.polyfit(train_x,train_y,d)
        model = np.poly1d(weights)
        predicted = model(testing_x)
        if predicted > y[end-1]:
            results.append('green')
        elif predicted < y[end-1]:
            results.append('red')
        else:
            results.append(df.loc[end-1,'label'])
        start += 1
        end += 1
    return results

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
index_year2 = df[df['Year']==2018].index.values.astype(int)
real_label = df[df['Year']==2018]['label'].values

d1 = 1; W1 = 12
results1 = linear_model(W1,d1,df)
accuracy1 = '%s%%'%round(sum(np.array(results1)==real_label)/len(results1)*100,2)
print('predicted label when d = 1 and W = 12\n', results1)
print('accuracy:',accuracy1,'\n')

d2 = 2; W2 = 9
results2 = linear_model(W2,d2,df)
accuracy2 = '%s%%'%round(sum(np.array(results2)==real_label)/len(results2)*100,2)
print('predicted label when d = 2 and W = 9\n', results2)
print('accuracy:',accuracy2,'\n')

d3 = 3; W3 = 11
results3 = linear_model(W3,d3,df)
accuracy3 = '%s%%'%round(sum(np.array(results3)==real_label)/len(results3)*100,2)
print('predicted label when d = 3 and W = 11\n', results3)
print('accuracy:',accuracy3,'\n')
