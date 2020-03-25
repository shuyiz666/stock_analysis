'''
assignment1: tips
question1: what is the average tip (as a percentage of meal cost) for for lunch and for dinner
'''
import os
import pandas as pd
import numpy as np

wd = os.getcwd()
ticker = 'tips'
input_dir = wd
file_name = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = 100.0 * df['tip']/df['total_bill']

# average tip for lunch and for dinner
average_tip_lunch = np.mean(df.tip[df.time == 'Lunch']/df.total_bill[df.time == 'Lunch'])
average_tip_dinner = np.mean(df.tip[df.time == 'Dinner']/df.total_bill[df.time == 'Dinner'])

print('the average tip for lunch is:','%s%%'%round(average_tip_lunch*100,2))
print('the average tip for dinner is:','%s%%'%round(average_tip_dinner*100,2))