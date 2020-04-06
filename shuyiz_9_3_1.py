'''
assignment3: random forest
question1: take N = 1,...,10 and d = 1,2,...,5. For each value of N and d construct a random tree classifier (use ”entropy” as splitting criteria - this is the default) use your year 1 labels as training set and compute the error rate for year 2. Plot your error rates and find the best combination of N and d.
'''
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
traning = df[df['Year']==2017]
testing = df[df['Year']==2018]

def random_forest(traning, testing, d, N):
    np.random.seed(1)
    X = traning[['mean_return', 'volatility']].values
    le = LabelEncoder()
    Y = le.fit_transform(traning['label'].values)
    Y_test = le.fit_transform(testing['label'].values)

    model = RandomForestClassifier(n_estimators=N,max_depth=d,criterion='entropy')
    model.fit(X,Y)

    new_instance = np.asmatrix(testing[['mean_return','volatility']].values)
    prediction = model.predict(new_instance)
    error_rate = np.mean(prediction != Y_test)
    return error_rate

error_rates = []
for N in range(1,11):
    for d in range(1,6):
        error_rate = random_forest(traning, testing, d, N)
        error_rates.append(error_rate)


plt.title('error rates in different N and d')
plt.xlabel('N', fontsize=14)
xlabels = [i for i in range(1,11) for j in range(1,6)]
plt.ylabel('d', fontsize=14)
ylabels = [i for i in range(1,6)]*10
plt.scatter(xlabels,ylabels, s=[i**4*20000 for i in error_rates])
plt.show()

plt.title('error rates in different N and K')
plt.xlabel('N')
plt.ylabel('error rate')
for d in range(1,6):
    error_rates_plot = []
    for N in range(1,11):
        error_rate2 = random_forest(traning, testing, d, N)
        error_rates_plot.append(error_rate2)
    plt.plot(list(range(1,11)),error_rates_plot)
plt.legend(['d = 1', 'd = 2','d = 3','d = 4','d = 5','d = 6'])
plt.show()

print('the smallest error rate is when N=5 d=3')