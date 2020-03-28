'''
assignment3: bakery dataset
question6: does this coffee shop make more money from selling drinks or from selling food
'''

import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

try:
    df = pd.read_csv(ticker_file)
    drinks = ['Coffee','Coffee granules','Coke','Hot chocolate','Juice','Mineral water','My-5 Fruit Shoot','Smoothies','Tea']
    unknown = ['Adjustment','Argentina Night','Art Tray','Bakewell','Basket','Christmas common','Fairy Doors','Farm House','Gift voucher','Hack the stack','Half slice Monster','Hearty & Seasonal','Keeping It Local','Mortimer','NONE','Nomad bag','Pick and Mix Bowls','Postcard','Scandinavian','Siblings','The BART','The Nomad','Tiffin','Tshirt',"Valentine's card"]
    df['type'] = 'food'
    for index,row in df.iterrows():
        if row['Item'] in drinks:
            df.loc[index,'type'] = 'drinks'
        elif row['Item'] in unknown:
            df.loc[index,'type'] = 'unknown'
    result = round(df.groupby('type')['Item_Price'].sum(),2)
    print(result,'\n')
    print('selling drinks make more money')
 

except Exception as e:
    print(e)
    print('failed to read data')
