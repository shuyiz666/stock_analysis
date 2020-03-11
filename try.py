import numpy as np
from sklearn.metrics import mean_squared_error ,r2_score
x = np.array([1, 2, 3, 4, 6, 8])
y = np.array([1, 1, 5, 8, 3, 5])
degree = 5
print(x,y,degree)
weights = np.polyfit(x,y, degree)
model = np.poly1d(weights)
predicted = model(x)
rmse = np.sqrt(mean_squared_error(y,predicted))
r2 = r2_score(y,predicted)