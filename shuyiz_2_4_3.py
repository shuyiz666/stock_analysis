'''
assigment 4: last digit distribution
question 3: compute the 4 error metrics
'''
import os
import pandas as pd
import statistics as sta

ticker = 'ZSAN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '.csv')
plot_dir = wd

try:
    df = pd.read_csv(ticker_file)
    data = []
    for i in range(2015, 2020):
        df_year = df[df['Year'] == i]
        digit = round(df_year['Open'],2).astype(str).str[-1]
        digit_fre = digit.value_counts(normalize=True).reset_index(name = 'freq')
        digit_fre.rename(columns = {'index':'digit'}, inplace = True)
        digit_fre['absolute error'] = digit_fre['freq'].map(lambda x:abs(x-0.1))
        max_error = max(digit_fre['absolute error'])
        median_error = sta.median(digit_fre['absolute error'])
        mean_error = sta.mean(digit_fre['absolute error'])
        RMSE_error = sta.mean(digit_fre['absolute error']**2)**0.5
        data.append([i, round(max_error,2), round(median_error,2), round(mean_error,2), round(RMSE_error,2)])
    title = ['year', 'max AE', 'median AE', 'mean AE', 'RMSE']
    result = pd.DataFrame(data, columns=title)
    print(result)
    print('')
    print('The error in 2018 and 2019 is smaller than other years.')
    print('2015 has the biggest max absolute error.')
    print('2017 has the smallest median absolute error.')
    print('2018 has the smallest RMSE.')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)