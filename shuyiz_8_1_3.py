'''
assignment1: tips
question3: when are tips highest (which day and time)?
'''
import os
import pandas as pd

wd = os.getcwd()
ticker = 'tips'
input_dir = wd
file_name = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = df['tip']/df['total_bill']

tip_each_day_time = df.groupby(['day','time'],as_index=False)['tip_percent'].agg({'tip_percent':'mean'})
tip_each_day_time['tip_percent'] = tip_each_day_time['tip_percent'].apply(lambda x:format(x,'.2%'))
highest_tip = tip_each_day_time[tip_each_day_time['tip_percent']==max(tip_each_day_time['tip_percent'])]

print(highest_tip) 
