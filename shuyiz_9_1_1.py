'''
assignment1: Discriminant Analysis
question1: what is the equation for linear and quadratic classifier found from year 1 data
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
    return lda_classifier

def qda(X,Y):
    qda_classifier = LQA()
    qda_classifier.fit(X,Y)
    return qda_classifier

equation_lda = lda(X,Y)
equation_qda = qda(X,Y)

print('equation for linear classifier is:')
print('y =',round(equation_lda.intercept_[0],2),'+(',round(equation_lda.coef_[0][0],2),')* x1 +(',round(equation_lda.coef_[0][1],2),')* x2')

