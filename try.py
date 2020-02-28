import numpy as np
import pandas as pd
import os
from collections import Counter

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)

training = df[df['Year'] == 2017]
X = training[['mean_return','volatility']].values
Labels = training['label'].values
print(Labels)
testing = df[df['Year'] == 2018]
new_x = testing[['mean_return','volatility']].values
