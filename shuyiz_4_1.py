'''
assignment1: Examine labels
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

df = pd.read_csv(ticker_file)
plt.scatter(df['mean_return'], df['volatility'],color=df['label'], s=20,alpha=0.5)
plt.title('mean vs. volatility')
plt.xlabel('mean')
plt.ylabel('volatility')
for a,b,c in zip(df['mean_return'],df['volatility'],df['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show()