'''
assigment 3: test normality returns
question 2:  compute the average of daily returns µ and compute the percentage of days with returns greater than µ and the proportion of days with returns less than µ
'''

import os
import pandas as pd

ticker = 'ZSAN'
input_dir = r'/Users/zhengshuyi'
ticker_file = os.path.join(input_dir, ticker + '.csv')
plot_dir = r'/Users/zhengshuyi'

try:
    df = pd.read_csv(ticker_file)
    data = []
    for i in range(2015,2020):
        df_year = df[df['Year'] == i]
        days = df_year.shape[0]
        mean = df_year['Return'].mean()
        less = df_year[df_year['Return'] < mean].shape[0]
        more = df_year[df_year['Return'] > mean].shape[0]
        data.append([i, days, round(mean,4), str(round(less/days*100,2))+'%',str(round(more/days*100,2))+'%'])
        positive = df_year[df_year['Return'] > 0].shape[0]
        negative = df_year[df_year['Return'] < 0].shape[0]
        if positive > negative:
            print('In '+ str(i) +', positive return days is more than negative return days')
        elif positive < negative:
            print('In ' + str(i) + ', negative return days is more than positive return days')
        else:
            print('In ' + str(i) + ', negative return days is equal to positive return days')
    title = ['year','trading days','u','%days < u','% days > u']
    result = pd.DataFrame(data,columns = title)
    print('It does not change from year to year')
    print('')
    print(result)
    print('')
    print('The percentage of days less than mean and greater than mean are very close.')
    print('In 2015 and 2016, the percentage of days greater than mean is greater than less than mean.')
    print('In 2017, 2018 and 2019, the percentage of days greater than mean is less than less than mean.')
    print('In 2017 and 2019, the average returns are positive while in other years are negative.')


except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)