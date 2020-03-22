'''
assignment2:Shapley Feature Explanations
question2:take IRIS dataset. In this set we have 3 flowers (Iris Ver- sicolor, Iris Setosa and Iris Virginica) and 4 features (sepal length, sepal width, petal length, petal width). We can as- sign multiple labels by training 3 distinct binary classifiers using ”one-vs-all” classification method. In this method, you take labels of one class as positive and the rest of the classesusing as negative. For example, take Iris Versicolor as one class and the remaining two (Iris Setosa and Iris Vir- ginica) as the second class. We then use a binary classifier using binary classification. (Making decisions means ap- plying all classifiers to an unseen sample x and predicting the label k for which the corresponding classifier reports confidence score). For each of the 3 flower types, construct a one-vs-all logistic regression classifier. For each of the of the 4 features (sepal length, sepal width, petal length, petal width) compute its marginal contributions ∆ to ac- curacy (split the dataset 50/50 into training and testing parts). Summarize your findings in a table (shown below) and discuss them
'''
import ssl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context

def logistic_regression(features,data):
    X = data[features].values
    le = LabelEncoder()
    Y = le.fit_transform(data['Class'].values)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5,
                                                        random_state=3)
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X_train, Y_train)

    prediction = log_reg_classifier.predict(X_test)
    accuracy = np.mean(prediction == Y_test)
    return accuracy


url = r'https://archive.ics.uci.edu/ml/'  + \
           r'machine-learning-databases/iris/iris.data'

data = pd.read_csv(url, names=['sepal-length', 'sepal-width',
                               'petal-length', 'petal-width', 'Class'])

features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

setosa = data.copy()
setosa['Class'] = setosa.Class.apply(lambda x: 1 if 'setosa' in x else 0)

sentota_accuracy = logistic_regression(features,setosa)

setosa_sepal_length = ['sepal-width', 'petal-length', 'petal-width']
setosa_sepal_length_accuracy = logistic_regression(setosa_sepal_length,setosa)

setosa_sepal_width = ['sepal-length', 'petal-length', 'petal-width']
setosa_sepal_width_accuracy = logistic_regression(setosa_sepal_width,setosa)

setosa_petal_length = ['sepal-length', 'sepal-width', 'petal-width']
setosa_petal_length_accuracy = logistic_regression(setosa_petal_length,setosa)

setosa_petal_width = ['sepal-length', 'sepal-width', 'petal-length']
setosa_petal_width_accuracy = logistic_regression(setosa_petal_width,setosa)


versicolor = data.copy()
versicolor['Class'] = versicolor.Class.apply(lambda x: 1 if 'versicolor' in x else 0)

versicolor_accuracy = logistic_regression(features,versicolor)

versicolor_sepal_length = ['sepal-width', 'petal-length', 'petal-width']
versicolor_sepal_length_accuracy = logistic_regression(versicolor_sepal_length,versicolor)

versicolor_sepal_length_sepal_width = ['sepal-length', 'petal-length', 'petal-width']
versicolor_sepal_length_sepal_width_accuracy = logistic_regression(versicolor_sepal_length_sepal_width,versicolor)

versicolor_sepal_length_petal_length = ['sepal-length', 'sepal-width', 'petal-width']
versicolor_sepal_length_petal_length_accuracy = logistic_regression(versicolor_sepal_length_petal_length,versicolor)

versicolor_sepal_length_petal_width = ['sepal-length', 'sepal-width', 'petal-length']
versicolor_sepal_length_petal_width_accuracy = logistic_regression(versicolor_sepal_length_petal_width,versicolor)

virginica = data.copy()
virginica['Class'] = virginica.Class.apply(lambda x: 1 if 'virginica' in x else 0)

virginica_accuracy = logistic_regression(features,virginica)

virginica_sepal_length = ['sepal-width', 'petal-length', 'petal-width']
virginica_sepal_length_accuracy = logistic_regression(virginica_sepal_length,virginica)

virginica_sepal_length_sepal_width = ['sepal-length', 'petal-length', 'petal-width']
virginica_sepal_length_sepal_width_accuracy = logistic_regression(virginica_sepal_length_sepal_width,virginica)

virginica_sepal_length_petal_length = ['sepal-length', 'sepal-width', 'petal-width']
virginica_sepal_length_petal_length_accuracy = logistic_regression(virginica_sepal_length_petal_length,virginica)

virginica_sepal_length_petal_width = ['sepal-length', 'sepal-width', 'petal-length']
virginica_sepal_length_petal_width_accuracy = logistic_regression(virginica_sepal_length_petal_width,virginica)

# data = {'Flower':['Versicolor','Setosa','Virginica'],
#         'sepal length':['%s%%'%(round((versicolor_accuracy-versicolor_sepal_length_accuracy)*100,2)),'%s%%'%(round((sentota_accuracy-setosa_sepal_length_accuracy)*100,2)),'%s%%'%(round((virginica_accuracy-virginica_sepal_length_accuracy)*100,2))],
#         'sepal width':['%s%%'%(round((versicolor_accuracy-versicolor_sepal_length_sepal_width_accuracy)*100,2)),'%s%%'%(round((sentota_accuracy-setosa_sepal_width_accuracy)*100,2)), '%s%%'%(round((virginica_accuracy-virginica_sepal_length_sepal_width_accuracy)*100,2))],
#         'petal length':['%s%%'%(round((versicolor_accuracy-versicolor_sepal_length_petal_length_accuracy)*100,2)),'%s%%'%(round((sentota_accuracy-setosa_petal_length_accuracy)*100,2)), '%s%%'%(round((virginica_accuracy-virginica_sepal_length_petal_length_accuracy)*100,2))],
#         'petal width':['%s%%'%(round((versicolor_accuracy-versicolor_sepal_length_petal_width_accuracy)*100,2)),'%s%%'%(round((sentota_accuracy-setosa_petal_width_accuracy)*100,2)), '%s%%'%(round((virginica_accuracy-virginica_sepal_length_petal_width_accuracy)*100,2))]}
#

data = {'Flower':['sepal length','sepal width','petal length', 'petal width'],
        'Versicolor':['%s%%'%(round((versicolor_accuracy-versicolor_sepal_length_accuracy)*100,2)),'%s%%'%(round((versicolor_accuracy-versicolor_sepal_length_sepal_width_accuracy)*100,2)),'%s%%'%(round((versicolor_accuracy-versicolor_sepal_length_petal_length_accuracy)*100,2)),'%s%%'%(round((versicolor_accuracy-versicolor_sepal_length_petal_width_accuracy)*100,2))],
        'Setosa':['%s%%'%(round((sentota_accuracy-setosa_sepal_length_accuracy)*100,2)),'%s%%'%(round((sentota_accuracy-setosa_sepal_width_accuracy)*100,2)), '%s%%'%(round((sentota_accuracy-setosa_petal_length_accuracy)*100,2)), '%s%%'%(round((sentota_accuracy-setosa_petal_width_accuracy)*100,2))],
        'Virginica':['%s%%'%(round((virginica_accuracy-virginica_sepal_length_accuracy)*100,2)),'%s%%'%(round((virginica_accuracy-virginica_sepal_length_sepal_width_accuracy)*100,2)), '%s%%'%(round((virginica_accuracy-virginica_sepal_length_petal_length_accuracy)*100,2)),'%s%%'%(round((virginica_accuracy-virginica_sepal_length_petal_width_accuracy)*100,2))]}

result = pd.DataFrame(data)
print(result)

print('sepal width has most influcenc to the accuracy while sepal length and length petal has least')
print('setosa has a very high accuracy and low delta which means each feature can decide its label')
