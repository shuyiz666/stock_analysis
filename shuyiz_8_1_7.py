'''
assignment1: tips
question7: assume that rows in the tips.csv file are arranged in time. Are tips increasing with time in each day?
'''
import os
import pandas as pd
import matplotlib.pyplot as plt

wd = os.getcwd()
ticker = 'tips'
input_dir = wd
file_name = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = df['tip']/df['total_bill']
plt.title('the relationship between days and tips')
plt.xlabel('day')
plt.ylabel('percentage tips')
plt.scatter(list(range(1,len(df)+1)),df['tip_percent'])
plt.show()

print('the tips are not increasing with time in each day')