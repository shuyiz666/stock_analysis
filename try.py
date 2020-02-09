import matplotlib.pylab as plt
import pandas as pd

scatter_positive = pd.DataFrame(columns=('w','k','value'))
scatter_negative = pd.DataFrame(columns=('w','k','value'))
k_range = [0.5,1]
row_positive, row_negative = 0, 0
x = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10]
i = 0
for w in range(10, 20):
    for k in k_range:
        if x[i] > 0:
            scatter_positive.loc[row_positive] = {'w': w, 'k': k, 'value': x[i]}
            row_positive += 1
        elif x[i] < 0:
            scatter_negative.loc[row_negative] = {'w': w, 'k': k, 'value': x[i]}
            row_negative += 1
        i += 1
print(scatter_negative)
plt.figure(figsize=(10, 5))
plt.title('Average P/L per transaction with change of W nad k')
plt.xlabel('W', fontsize=14)
plt.ylabel('k', fontsize=14)
plt.scatter(scatter_positive['w'], scatter_positive['k'], s=scatter_positive['value'], c='green')
plt.scatter(scatter_negative['w'], scatter_negative['k'], s=-(scatter_negative['value']), c='red')
plt.legend(['profit', 'loss'])
plt.show()