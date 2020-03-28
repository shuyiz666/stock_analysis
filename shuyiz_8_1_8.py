'''
assignment1: tips
question8: is there any difference in correlation between tip amounts from smokers and non-smokers?
'''
import os
import pandas as pd


wd = os.getcwd()
ticker = 'tips'
input_dir = wd
file_name = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = df['tip']/df['total_bill']

smokers = df.groupby(['smoker'])['tip_percent'].mean()

if smokers['No'] > smokers['Yes']:
    print("Non-smokers pay more tips.")
elif smokers['No'] < smokers['Yes']:
    print("Smokers pay more tips.")
else:
    print("Smokers and non-smokers pay equal tips")
 
