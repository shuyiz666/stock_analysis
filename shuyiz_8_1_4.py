'''
assignment1: tips
question4: compute the correlation between meal prices and tips
'''
import os
import pandas as pd
from scipy.stats import pearsonr

wd = os.getcwd()
ticker = 'tips'
input_dir = wd
file_name = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = df['tip']/df['total_bill']
df['meal_price'] = df['total_bill']-df['tip']
corr, _ = pearsonr(df['meal_price'], df['tip_percent'])
print('Pearsons correlation: %.2f'%corr) 
