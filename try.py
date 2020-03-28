import os
import pandas as pd
import numpy as np
import math

df = pd.read_csv('jia_mix.csv')
print(df.iloc[:,1:23])
df.iloc[:,1:24] = df.iloc[:,1:24].apply(lambda x:x if x==0 else np.log(x))
print(df)