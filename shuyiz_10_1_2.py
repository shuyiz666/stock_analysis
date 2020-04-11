'''
assignment1: SVM
question2:  compute the confusion matrix for year 2
'''
import os
import pandas as pd
from sklearn import svm
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

svm_classifier = svm.SVC(kernel = 'linear')
svm_classifier.fit(X,Y)
new_x = scaler.transform(testing[['mean_return','volatility']].values)
predicted = svm_classifier.predict(new_x)
cm = confusion_matrix(testing[['label']].values.ravel(), predicted)
print('confusion matrix is:\n',cm,'\n')
