'''
assigment 4: last digit distribution
question 2: what is the least frequent digit
'''
import os
import pandas as pd

ticker = 'ZSAN'
input_dir = r'/Users/zhengshuyi'
ticker_file = os.path.join(input_dir, ticker + '.csv')
plot_dir = r'/Users/zhengshuyi'

try:
    df = pd.read_csv(ticker_file)
    digit = round(df['Open'],2).astype(str).str[-1]
    digit_fre = round(digit.value_counts(normalize=True),2).reset_index(name = 'freq')
    digit_fre.rename(columns = {'index':'digit'}, inplace = True)
    print(digit_fre)
    print('')
    print('The least frequent digit is 9')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)