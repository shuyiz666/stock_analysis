'''
assignment1: Discriminant Analysis
question3: compute the confusion matrix for year 2 for each classifier
'''

import os
import pandas as pd
from sklearn . discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn . discriminant_analysis import QuadraticDiscriminantAnalysis as LQA
from sklearn . preprocessing import StandardScaler
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

cm_lda = confusion_matrix(testing['label'].values, lda)
cm_qda = confusion_matrix(testing['label'].values, qda)

print('the confusion matrix for linear classifier is:\n',cm_lda)
print('the confusion matrix for quadratic classifier is:\n',cm_qda)