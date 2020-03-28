'''
assignment3: Linear Separability
question2: take this line and use it to assign labels for year 2
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

df = pd.read_csv(ticker_file)
df = df[df['Year']==2018]
df['predict_label'] = 'NULL'

# assign year2
for index, row in df.iterrows():
    if row['mean_return']*15-15 >= row['volatility']:
        df.loc[index,'predict_label'] = 'green'
    else:
        df.loc[index,'predict_label'] = 'red'
print(df)

# plot year2
plt.scatter(df['mean_return'], df['volatility'],color=df['predict_label'], s=20,alpha=0.5)
plt.plot([1,11/3],[0,40],color = 'black', ls = 'dotted')
plt.title('predict label with equation in year2')
plt.xlabel('mean')
plt.ylabel('volatility')
for a,b,c in zip(df['mean_return'],df['volatility'],df['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show() 
