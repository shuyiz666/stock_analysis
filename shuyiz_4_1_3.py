'''
assignment1: Examine labels
question3:  do patterns repeat from year 1 to year 2
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
year1 = df[df['Year']==2017]
year2 = df[df['Year']==2018]
# year 1
plt.scatter(year1['mean_return'], year1['volatility'],color=year1['label'], s=20,alpha=0.5)
plt.title('year1 mean vs. volatility')
plt.xlabel('mean %')
plt.ylabel('volatility %')
for a,b,c in zip(year1['mean_return'],year1['volatility'],year1['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show()

# year 2
plt.scatter(year2['mean_return'], year2['volatility'],color=year2['label'], s=20,alpha=0.5)
plt.title('year2 mean vs. volatility')
plt.xlabel('mean %')
plt.ylabel('volatility %')
for a,b,c in zip(year2['mean_return'],year2['volatility'],year2['Week_Number']):
    plt.text(a,b+0.1,c,ha = 'center',va = 'bottom',fontsize=7)
plt.show()

print('Yes, patterns repeat from year 1 to year 2')
