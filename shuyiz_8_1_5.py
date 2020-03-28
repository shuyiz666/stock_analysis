'''
assignment1: tips
question5: is there any relationship between tips and size of the group
'''
import os
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

wd = os.getcwd()
ticker = 'tips'
input_dir = wd
file_name = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = df['tip']/df['total_bill']
# correlation of percentage of tips and size of group
corr, _ = pearsonr(df['tip_percent'], df['size'])
print('the Pearsons correlation of size of group and percentage of tips is %.2f'%corr,'which means there is no strong relationship betweeen them')


# plot percentage of tips and size of group
plt.title('the relationship between size of group and percentage of tips')
plt.xlabel('size of group')
plt.ylabel('percentage of tips')
plt.scatter(df['size'],df['tip_percent'])
plt.show()
print('It seems that the percentage of tips in different size of group are quite similar except some particular situation in size 2')

# y:tips per person x:size of the group
df['tip_per_persion'] = df['tip']/df['size']
plt.title('the relationship between size of group and average tips per person')
plt.xlabel('size of group')
plt.ylabel('average tips per person')
plt.scatter(df['size'],df['tip_per_persion'])
plt.show()
print('People with group size 2-4 are tend to pay more tips per person') 
