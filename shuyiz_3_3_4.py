'''
assignment3: bakery dataset
question4: How many barristas do you need for each day of the week
'''

import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

try:
    df = pd.read_csv(ticker_file)
    result = df.groupby(['Weekday','Year','Month','Day'])['Transaction'].count().groupby('Weekday').max()
    print(result/50,'\n')

    print('5 barristas are needed for Monday')
    print('4 barrista is needed for Tuesday')
    print('4 barristas are needed for Wednesday')
    print('4 barrista is needed for Thursday')
    print('4 barristas are needed for Friday')
    print('7 barristas are needed for Saturday')
    print('5 barristas are needed for Sunday')

except Exception as e:
    print(e)
    print('failed to read data')