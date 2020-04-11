'''
assignment3: random forest
question2: using the optimal values from year 1, compute the confusion matrix for year 2
'''
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
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
    X = traning[['mean_return', 'volatility']].values
    le = LabelEncoder()
    Y = le.fit_transform(traning['label'].values)
    Y_test = le.fit_transform(testing['label'].values)

    model = RandomForestClassifier(n_estimators=N,max_depth=d,criterion='entropy')
    model.fit(X,Y)

    new_instance = np.asmatrix(testing[['mean_return','volatility']].values)
    prediction = model.predict(new_instance)
    return prediction,Y_test


prediction,Y_test = random_forest(traning, testing, 3, 5)
accuracy = sum(prediction == Y_test)/len(Y_test)
cm = confusion_matrix(Y_test, prediction)
print('the confusion matrix for random forest is:\n',cm)
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
TPR = TP/(TP+FN)
TNR = TN/(TN+FP)