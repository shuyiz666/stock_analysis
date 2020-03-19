import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import mean_squared_error,r2_score
x = np.array([1, 2, 3, 4, 6, 8])
y = np.array([1, 1, 5, 8, 3, 5])
degree = 2
weights = np.polyfit(x,y, degree)
model = np.poly1d(weights)
predicted = model(x)
print(y)
r2 = r2_score(np.array([1]),np.array([2]))
print(r2)