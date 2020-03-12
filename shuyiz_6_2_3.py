'''
assignment2: Weekly Trading with Linear Models
question3: compute confusion matrices (for each d) for year 2
'''
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

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


for d in range(1,4):
    for W in range(5,13):
        results = linear_model(W, d, df)
        cm = confusion_matrix(real_label, results)
        print('the confusion matrics for d =',d,' and W =',W,':\n',cm,'\n')


