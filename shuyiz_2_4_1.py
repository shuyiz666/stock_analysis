'''
assigment 4: last digit distribution
question 1: what is the most frequent digit
'''
import os
import pandas as pd

ticker = 'ZSAN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
plot_dir = wd

try:
    df = pd.read_csv(ticker_file)
    digit = round(df['Open'],2).astype(str).str[-1]
    digit_fre = round(digit.value_counts(normalize=True),2).reset_index(name = 'freq')
    digit_fre.rename(columns = {'index':'digit'}, inplace = True)
    print(digit_fre)
    print('')
    print('The most frequent digit is 0.')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)