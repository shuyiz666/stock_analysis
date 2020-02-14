'''
assignment3: bakery dataset
question1: what is the busiest
'''

import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

try:
    df = pd.read_csv(ticker_file)
    print(df.groupby('Hour')['Transaction'].nunique(),'\n')
    print(df.groupby('')['Transaction'].nunique(),'\n')
    print(df.groupby('Period')['Transaction'].nunique(),'\n')

    print('(a) 11 is the busiest hour with 1445 transactions')
    print('(b) Saturday is the busiest day of the week with 2068 transactions')
    print('(c) afternoon is the busiest period with 5307 transactions')


except Exception as e:
    print(e)
    print('failed to read data')