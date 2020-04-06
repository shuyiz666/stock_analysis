'''
assignment1: Discriminant Analysis
question4: what is true positive rate (sensitivity or recall) and true negative rate (specificity) for year 2?
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
TN_lda = cm_lda[0][0]
FN_lda = cm_lda[1][0]
TP_lda = cm_lda[1][1]
FP_lda = cm_lda[0][1]
TPR_lda = TP_lda/(TP_lda+FN_lda)
TNR_lda = TN_lda/(TN_lda+FP_lda)
print('true positive rate for linear classifier is','%s%%'%round(TPR_lda*100,2))
print('true negative rate for quadratic classifier is','%s%%'%round(TNR_lda*100,2))

cm_qda = confusion_matrix(testing['label'].values, qda)
TN_qda = cm_qda[0][0]
FN_qda = cm_qda[1][0]
TP_qda = cm_qda[1][1]
FP_qda = cm_qda[0][1]
TPR_qda = TP_qda/(TP_qda+FN_qda)
TNR_qda = TN_qda/(TN_qda+FP_qda)
print('true positive rate for linear classifier is','%s%%'%round(TPR_qda*100,2))
print('true negative rate for quadratic classifier is','%s%%'%round(TNR_qda*100,2))