'''
assignment3: Linear Separability
question1: take year 1 and examine the plot of your labels. Construct a reduced dataset by removing some green and red points so thatyou can draw a line separating the points. Compute the equation of such a line (many solutiuons are possible)
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

# plot before removing points
df = pd.read_csv(ticker_file)
df = df[df['Year']==2017]
plt.scatter(df['mean_return'], df['volatility'],color=df['label'], s=20,alpha=0.5)
plt.title('plot before removing points')
plt.xlabel('mean')
plt.ylabel('volatility')
for a,b,c in zip(df['mean_return'],df['volatility'],df['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show()

# plot after removing points
df_remove = df[~df.Week_Number.isin([4,12,21,22])]
plt.scatter(df_remove['mean_return'], df_remove['volatility'],color=df_remove['label'], s=20,alpha=0.5)
plt.plot([1,2.2],[0,18],color = 'black', ls = 'dotted')
plt.title('plot after removing points')
plt.xlabel('mean')
plt.ylabel('volatility')
for a,b,c in zip(df_remove['mean_return'],df_remove['volatility'],df_remove['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show()
 
print('the equation is: y = 15*x-15')
