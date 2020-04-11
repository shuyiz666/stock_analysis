'''
assignment1: SVM
question5: implement polynomial SVM for degree 2 and compute its accuracy? Is it better than linear SVM?
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

svm_classifier = svm.SVC(kernel = 'poly',degree=2)
svm_classifier.fit(X,Y)
new_x = scaler.transform(testing[['mean_return','volatility']].values)
predicted = svm_classifier.predict(new_x)
accuracy = sum(predicted == testing[['label']].values.ravel())/len(predicted)
print('the accuracy of polynomial SVM is:','%s%%'%(round(accuracy*100,2)))
print('it is worse than linear SVM')