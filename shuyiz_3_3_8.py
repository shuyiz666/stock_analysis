'''
assignment3: bakery dataset
question8:  what are the bottom 5 least popular items for each day of the week
'''
import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

try:
    df = pd.read_csv(ticker_file)
    result = df.groupby(['Weekday'])['Item'].value_counts().rename('count').reset_index()
    print(result.groupby('Weekday').tail(5),'\n')
    print('The least popular items are quiet different from day to day.')
    print('Drinking chocolate spoons is least popular on Monday, Thursday, Sunday.')
    print('Victorian Sponge is least popular on Tuesday, Wednesday, Saturday.')

except Exception as e:
    print(e)
    print('failed to read data')
    
