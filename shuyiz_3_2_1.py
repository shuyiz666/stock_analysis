'''
assignment2: retail bernford
question1: plot 3 histograms
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
    print('true value:')
    print(real_distribution)
    print('')
    print('Model1 equal weight: ')
    print(equal_weight)
    print('')
    print("Model2 Bernford's law: ")
    print(Bernford)
    total_width, n = 0.8, 3
    width = total_width/n
    x2 = [x + width for x in equal_weight['digit']]
    x3 = [x + 2*width for x in equal_weight['digit']]
    plt.bar(equal_weight['digit'],equal_weight['freq'], width = width,label='equal_weight')
    plt.bar(x2,Bernford['freq'], width = width, label='Bernford')
    plt.bar(x3,real_distribution['freq'], width = width,label='real_distribution')
    plt.title('frequencies for real distribution, equal-weight and Bernford')
    plt.legend()
    plt.xlabel('digit')
    plt.ylabel('frequency')
    plt.show()


except Exception as e:
    print(e)
    print('failed to read data')
    
