'''
assignment2: retail bernford
question3: compute RMSE
'''

import os
import pandas as pd
import math
import statistics as sta

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
    equal_weight_merge['absolute errors square'] = (equal_weight_merge['freq_x'] - equal_weight_merge['freq_y'])**2
    Bernford_merge = pd.merge(real_distribution, Bernford, on = 'digit', how = 'outer')
    Bernford_merge['absolute errors square'] = (Bernford_merge['freq_x'] - Bernford_merge['freq_y'])**2
    print('RMSE for Model 1 is: ',sta.mean(equal_weight_merge['absolute errors square'])**0.5)
    print('RMSE for Model 2 is: ',sta.mean(Bernford_merge['absolute errors square'])**0.5)
    print('Model 1:')
    print(equal_weight_merge)
    print('')
    print('Model 2:')
    print(Bernford_merge)

except Exception as e:
    print(e)
    print('failed to read data')