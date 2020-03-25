'''
assignment1: tips
question8: is there any difference in correlation between tip amounts from smokers and non-smokers?
'''
import os
import pandas as pd


wd = os.getcwd()
ticker = 'tips'
input_dir = wd
file_name = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = df['tip']/df['total_bill']

smokers = df.groupby(['smoker'])['tip_percent'].mean()
if smokers > :
#     print("Non-smokers pay larger tips.")
# elif mean_tip_smoke_no < mean_tip_smoke_yes:
#     print("Smokers pay larger tips.")
# else:
#     print("Smokers and non-smokers pay equal tips")

# plt.title('the relationship between days and tips')
# plt.xlabel('day')
# plt.ylabel('percentage tips')
# plt.scatter(list(range(1,len(df)+1)),df['tip_percent'])
# plt.show()
#
# print('the tips are not increasing with time in each day')