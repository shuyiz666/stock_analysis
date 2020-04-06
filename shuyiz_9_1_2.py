'''
assignment1: Discriminant Analysis
question2: what is the accuracy for year 2 for each classifier. Which classifier is ”better”
'''

import os
import pandas as pd
from sklearn . discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn . discriminant_analysis import QuadraticDiscriminantAnalysis as LQA
from sklearn . preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)
traning = df[df['Year']==2017]
testing = df[df['Year']==2018]

X = traning[['mean_return','volatility']].values
scaler = StandardScaler()
scaler .fit(X)

X = scaler.transform(X)
Y = traning[['label']].values.ravel()

def lda(X,Y):
    lda_classifier = LDA(n_components = 2)
    lda_classifier.fit(X,Y)
    new_instance = scaler.transform(testing[['mean_return', 'volatility']].values)
    predicted = lda_classifier.predict(new_instance)
    return predicted

def qda(X,Y):
    qda_classifier = LQA()
    qda_classifier.fit(X,Y)
    new_instance = scaler.transform(testing[['mean_return', 'volatility']].values)
    predicted = qda_classifier.predict(new_instance)
    return predicted

lda = lda(X,Y)
qda = qda(X,Y)

accuracy_lda = sum(testing['label'].values==lda)/len(lda)
accuracy_qda = sum(testing['label'].values==qda)/len(qda)

print('the accuracy for year2 for linear classifier is','%s%%'%round(accuracy_lda*100,2))
print('the accuracy for year2 for quadratic classifier is','%s%%'%round(accuracy_qda*100,2))

if accuracy_lda > accuracy_qda:
    print('linear classifier is better')
elif accuracy_lda < accuracy_qda:
    print('qudratic classifier is better')
else:
    print('the accuracy for two classifiers are equal')