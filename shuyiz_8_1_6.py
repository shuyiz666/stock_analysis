'''
assignment1: tips
question6: what percentage of people are smoking
'''
import os
import pandas as pd

wd = os.getcwd()
ticker = 'tips'
input_dir = wd
file_name = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(file_name)

df['smoke_people'] = df.smoker.apply(lambda x:1 if 'Yes' in x else 0)
smoke = sum(df['smoke_people']*df['size'])
total = sum(df['size'])
print('%s%%'%round(smoke/total*100,2))
 
