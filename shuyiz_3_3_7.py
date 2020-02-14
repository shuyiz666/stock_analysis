'''
assignment3: bakery dataset
question7:  what are the top 5 most popular items for each day of the week
'''
import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

try:
    df = pd.read_csv(ticker_file)
    result = df.groupby(['Weekday'])['Item'].value_counts().rename('count').reset_index()
    print(result.groupby('Weekday').head(5),'\n')
    print('Coffee, bread, tea are top3 every day of the week.')
    print('Sandwich is popular on Monday, Friday.')
    print('Cake is not popular on Monday.')

except Exception as e:
    print(e)
    print('failed to read data')