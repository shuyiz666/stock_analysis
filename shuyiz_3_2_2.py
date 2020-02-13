'''
assignment2: retail bernford
question2: plot relative errors
'''

import os
import pandas as pd
import math
import matplotlib.pyplot as plt

wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, 'online_retail.csv')

try:
    df = pd.read_csv(ticker_file)
    digit = df['UnitPrice'][df['UnitPrice'] >= 1].astype(str).str[0]
    real_distribution = round(digit.value_counts(normalize=True),2).reset_index(name='freq')
    real_distribution.rename(columns={'index': 'digit'}, inplace=True)
    equal_weight_data = {'digit':[1,2,3,4,5,6,7,8,9],
                         'freq':[0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11]}
    equal_weight = pd.DataFrame(equal_weight_data)
    Bernford_data = {'digit':[1,2,3,4,5,6,7,8,9],
                     'freq': [round(math.log10(1+1/1),2),round(math.log10(1+1/2),2),round(math.log10(1+1/3),2),round(math.log10(1+1/4),2),round(math.log10(1+1/5),2),round(math.log10(1+1/6),2),round(math.log10(1+1/7),2),round(math.log10(1+1/8),2),round(math.log10(1+1/9),2)]}
    Bernford = pd.DataFrame(Bernford_data)

    real_distribution['digit'] = real_distribution['digit'].astype(int)
    equal_weight['digit'] = equal_weight['digit'].astype(int)
    Bernford['digit'] = Bernford['digit'].astype(int)

    equal_weight_merge = pd.merge(real_distribution,equal_weight, on = 'digit', how = 'outer')
    equal_weight_merge['relative errors'] = abs(equal_weight_merge['freq_x'] - equal_weight_merge['freq_y'])/equal_weight_merge['freq_x']
    equal_weight_merge['relative errors'] = equal_weight_merge['relative errors'].apply(lambda x: format(x, '.2%'))
    Bernford_merge = pd.merge(real_distribution, Bernford, on = 'digit', how = 'outer')
    Bernford_merge['relative errors'] = abs(Bernford_merge['freq_x'] - Bernford_merge['freq_y'])/Bernford_merge['freq_x']
    Bernford_merge['relative errors'] = Bernford_merge['relative errors'].apply(lambda x: format(x, '.2%'))

    print('Model 1:')
    print(equal_weight_merge)
    print('')
    print('Model 2:')
    print(Bernford_merge)

    total_width, n = 0.8, 2
    width = total_width/n
    x2 = [x + width for x in equal_weight_merge['digit']]
    plt.bar(equal_weight_merge['digit'],equal_weight_merge['relative errors'], width = width,label='Model 1')
    plt.bar(x2,Bernford_merge['relative errors'], width = width, label='Model 2')
    plt.title('relative errors for Models 1 and 2')
    plt.legend()
    plt.xlabel('digit')
    plt.ylabel('relative errors')
    plt.show()


except Exception as e:
    print(e)
    print('failed to read data')