'''
assignment1: Examine labels
question1: Are there any ”obvious” patterns? For example, for higher σ and μ are there more green points?
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')

df = pd.read_csv(ticker_file)
df['mean_return'] = df['mean_return'].round(2)
df['volatility'] = df['volatility'].round(2)

plt.scatter(df['mean_return'], df['volatility'],color=df['label'], s=20,alpha=0.5)
plt.title('mean vs. volatility')
plt.xlabel('mean %')
plt.ylabel('volatility %')
for a,b,c in zip(df['mean_return'],df['volatility'],df['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show()
print('Yes, there are obvious patterns. For higher mu, there are more green points.')
