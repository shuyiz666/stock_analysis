'''
assignment3: bakery dataset
question2: what is the most profitable
'''

import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

try:
    df = pd.read_csv(ticker_file)
    print(df.groupby('Hour')['Item_Price'].sum(),'\n')
    print(df.groupby('Weekday')['Item_Price'].sum(),'\n')
    print(df.groupby('Period')['Item_Price'].sum(),'\n')

    print('(a) 11 is the most profitable hour with profit 21453.44')
    print('(b) Saturday is the most profitable day of the week with profit 31531.83')
    print('(c) afternoon is the the most profitable period with profit 81299.97')


except Exception as e:
    print(e)
    print('failed to read data')