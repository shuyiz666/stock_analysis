import numpy as np
import pandas as pd
import os
from collections import Counter

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)

training = df[df['Year'] == 2017]
X = training[['mean_return','volatility']].values
Labels = training['label'].values
testing = df[df['Year'] == 2018]
new_x = testing[['mean_return','volatility']].values

# self.X 是df 只有mean和sigma， new_x是testing只有mean和sigma
df_dists = pd.DataFrame(columns = ['label','distance'])
# self.X 是df 只有mean和sigma， new_x是testing只有mean和sigma
for i in new_x:
    for j in range(len(X)):
        distance = np.linalg.norm(i - X[j], ord=1)
        df_dists.loc[j] = [Labels[j],distance]
    S = df_dists.sort_values(by='distance',ascending=True)
    toplabel = S['label'][0:5]
    predict_label = Counter(toplabel).most_common(1)[0][0]
    labels.append(predict_label)
    # dists = sorted(dists[1])[5]
    # print(dists)
    # topK = Counter(dists).most_common(5)
    # print(topK)
    # print(topdis)
    # for i in topdis:
    #     print(i.keys())
    # toplabel = []
    # for dis in topdis:
    #     print(dists.index(dis))
    #     toplabel.append(Labels[dists.index(dis)])
    # print(toplabel)