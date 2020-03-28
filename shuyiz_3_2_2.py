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

    # relative error for model 1
    # real_distribution and model 1
    real_distribution_merge = pd.merge(real_distribution,equal_weight, on = 'digit', how = 'outer')
    # freq_x: real_distribution, freq_y: model 1
    real_distribution_merge['relative errors'] = abs(real_distribution_merge['freq_x'] - real_distribution_merge['freq_y'])/real_distribution_merge['freq_y']

    # equal_weight_distribution and model 1
    equal_weight_merge = pd.merge(equal_weight, equal_weight, on = 'digit', how = 'outer')
    # freq_x: equal_weight_distribution, freq_y: model 1
    equal_weight_merge['relative errors'] = abs(equal_weight_merge['freq_x'] - equal_weight_merge['freq_y']) / equal_weight_merge['freq_y']

    # bernford distribution and model 1
    Bernford_merge = pd.merge(Bernford, equal_weight, on = 'digit', how = 'outer')
    # freq_x: Bernford, freq_y: model 1
    Bernford_merge['relative errors'] = abs(Bernford_merge['freq_x'] - Bernford_merge['freq_y'])/Bernford_merge['freq_y']

    print('Relative error for model 1 as compared of frequencies with real distribution:')
    print(real_distribution_merge,'\n')
    print('Relative error for model 1 as compared of frequencies with equal-weight distribution:')
    print(equal_weight_merge,'\n')
    print('Relative error for model 1 as compared of frequencies with Bernford distribution:')
    print(Bernford_merge,'\n')

    total_width, n = 0.8, 3
    width = total_width/n
    x2 = [x + width for x in equal_weight_merge['digit']]
    x3 = [x + width*2 for x in Bernford_merge['digit']]
    plt.bar(real_distribution_merge['digit'],real_distribution_merge['relative errors'], width = width,label='compared with real distribution')
    plt.bar(x2,equal_weight_merge['relative errors'], width = width, label='compared equal-weight distribution')
    plt.bar(x3,Bernford_merge['relative errors'], width = width, label='compared with Bernford distribution')
    plt.title('relative errors for Model 1')
    plt.legend()
    plt.xlabel('digit')
    plt.ylabel('relative errors')
    plt.show()

    # relative error for model 2
    # real_distribution and model 2
    real_distribution_merge2 = pd.merge(real_distribution,Bernford, on = 'digit', how = 'outer')
    # freq_x: real_distribution, freq_y: model 2
    real_distribution_merge2['relative errors'] = abs(real_distribution_merge2['freq_x'] - real_distribution_merge2['freq_y'])/real_distribution_merge2['freq_y']

    # equal_weight_distribution and model 2
    equal_weight_merge2 = pd.merge(equal_weight, Bernford, on = 'digit', how = 'outer')
    # freq_x: equal_weight_distribution, freq_y: model 1
    equal_weight_merge2['relative errors'] = abs(equal_weight_merge2['freq_x'] - equal_weight_merge2['freq_y']) / equal_weight_merge2['freq_y']

    # bernford distribution and model 1
    Bernford_merge2 = pd.merge(Bernford, Bernford, on = 'digit', how = 'outer')
    # freq_x: Bernford, freq_y: model 2
    Bernford_merge2['relative errors'] = abs(Bernford_merge2['freq_x'] - Bernford_merge2['freq_y'])/Bernford_merge2['freq_y']

    print('Relative error for model 2 as compared of frequencies with real distribution:')
    print(real_distribution_merge2,'\n')
    print('Relative error for model 2 as compared of frequencies with equal-weight distribution:')
    print(equal_weight_merge2,'\n')
    print('Relative error for model 2 as compared of frequencies with Bernford distribution:')
    print(Bernford_merge2,'\n')

    plt.bar(real_distribution_merge2['digit'],real_distribution_merge2['relative errors'], width = width,label='compared with real distribution')
    plt.bar(x2,equal_weight_merge2['relative errors'], width = width, label='compared equal-weight distribution')
    plt.bar(x3,Bernford_merge2['relative errors'], width = width, label='compared with Bernford distribution')
    plt.title('relative errors for Model 2')
    plt.legend()
    plt.xlabel('digit')
    plt.ylabel('relative errors')
    plt.show()

except Exception as e:
    print(e)
    print('failed to read data')
    
