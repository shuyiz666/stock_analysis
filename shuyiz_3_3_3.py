'''
assignment3: bakery dataset
question3: what is the most and least popular item
'''

import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

try:
    df = pd.read_csv(ticker_file)
    result = df.groupby('Item')['Item_Price'].count()
    pd.options.display.max_rows = None
    print(result.sort_values(),'\n')

    print('the most popular item is coffee with 5471 record')
    print('the least popular item are Adjustment, Chicken sand, Olum & polenta, Polenta, Bacon, Gift voucher, The BART, Raw bars with only 1 record')


except Exception as e:
    print(e)
    print('failed to read data')
    
