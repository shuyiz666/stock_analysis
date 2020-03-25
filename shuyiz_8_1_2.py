'''
assignment1: tips
question2: what is average tip for each day of the week (as a percentage of meal cost)?
'''
import os
import pandas as pd

wd = os.getcwd()
ticker = 'tips'
input_dir = wd
file_name = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = df['tip']/df['total_bill']

average_tip = df.groupby(['day'],as_index=False)['tip_percent'].agg({'tip_percent':'mean'})
average_tip['tip_percent'] = average_tip['tip_percent'].apply(lambda x:format(x,'.2%'))

print(average_tip)