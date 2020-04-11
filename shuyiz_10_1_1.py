'''
assignment1: SVM
question1:  implement a linear SVM. What is the accuracy of your SVM for year 2
'''
import os
import pandas as pd
from sklearn import svm
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

svm_classifier = svm.SVC(kernel = 'linear')
svm_classifier.fit(X,Y)
new_x = scaler.transform(testing[['mean_return','volatility']].values)
predicted = svm_classifier.predict(new_x)
accuracy = sum(predicted == testing[['label']].values.ravel())/len(predicted)
print('the accuracy of linear SVM is:','%s%%'%(round(accuracy*100,2)))