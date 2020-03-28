'''
assignment1: Shapley Feature Explanations
question1: compute the contributions of μ and σ for logistic regression, Euclidean kNN and (degree 1) linear model. Summarize them in a table and discuss your findings
'''
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def logistic_regression(train,label,test):
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(train,label)
    test2 = scaler.transform(test)
    prediction = log_reg_classifier.predict(test2)
    return prediction

def KNN(train, label, test):
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(train, np.ravel(label))
    prediction = knn_classifier.predict(test)
    return prediction

# def linear_model(train, y, test, previous_price, previous_label):
#     model = LinearRegression().fit(train,y)
#     predicted = model.predict(test)
#     prediction = []
#     for i in range(len(predicted)):
#         if predicted[i] > previous_price[i]:
#             prediction.append('green')
#         elif predicted[i] < previous_price[i]:
#             prediction.append('red')
#         else:
#             prediction.append(previous_label[i])
#     prediction = np.array(prediction)
#     return prediction

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)

training = df[df['Year']==2017]
testing = df[df['Year']==2018]
train = training[['mean_return', 'volatility']].values
train_mu = training[['mean_return']].values
train_sigma = training[['volatility']].values

label = training['label'].values

# linear_y = training['Adj Close'].values
# previous_price = df.loc[df[df['Year']==2018].index[0]-1:len(df),'Adj Close'].values
# previous_label = df.loc[df[df['Year']==2018].index[0]-1:len(df),'label'].values

test = testing[['mean_return', 'volatility']].values
test_mu = testing[['mean_return']].values
test_sigma =  testing[['volatility']].values

real = testing['label'].values

total_logistic_regression = logistic_regression(train,label,test)
delta1_logistic_regression = logistic_regression(train_sigma,label,test_sigma)
delta2_logistic_regression = logistic_regression(train_mu,label,test_mu)


accuracy_logistic_regression = sum(total_logistic_regression == real)/len(real)
accuracy_delta1_logistic_regression = sum(delta1_logistic_regression == real)/len(real)
accuracy_delta2_logistic_regression = sum(delta2_logistic_regression == real)/len(real)

total_KNN = KNN(train, label, test)
delta1_KNN = KNN(train_sigma,label,test_sigma)
delta2_KNN = KNN(train_mu,label,test_mu)

accuracy_KNN = sum(total_KNN == real)/len(real)
accuracy_delta1_KNN = sum(delta1_KNN == real)/len(real)
accuracy_delta2_KNN = sum(delta2_KNN == real)/len(real)

# total_linear_model = linear_model(train, linear_y, test, previous_price, previous_label)
# delta1_linear_model = linear_model(train_sigma,linear_y,test_sigma ,previous_price, previous_label)
# delta2_linear_model = linear_model(train_mu,linear_y,test_mu,previous_price, previous_label)

# accuracy_linear_model = sum(total_linear_model == real)/len(real)
# accuracy_delta1_linear_model = sum(delta1_linear_model == real)/len(real)
# accuracy_delta2_linear_model = sum(delta2_linear_model == real)/len(real)

data = {'model':['logistic regression','KNN'],
        'accuracy':['%s%%'%(round(accuracy_logistic_regression*100,2)),'%s%%'%(round(accuracy_KNN*100,2))],
        'delta1':['%s%%'%(round((accuracy_logistic_regression-accuracy_delta1_logistic_regression)*100,2)),'%s%%'%(round((accuracy_KNN-accuracy_delta1_KNN)*100,2))],
        'delta2':['%s%%'%(round((accuracy_logistic_regression-accuracy_delta2_logistic_regression)*100,2)),'%s%%'%(round((accuracy_KNN-accuracy_delta2_KNN)*100,2))]}
result = pd.DataFrame(data)
print(result)

print('it seems mu has larger influence to accuracy than sigma since delta1 is bigger than delta2 in all models')
print('KNN has biggest change in three models when remove one feature') 
