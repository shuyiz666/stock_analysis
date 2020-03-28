'''
assignment3: bakery dataset
question5: What is the average price of a drink and a food item
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
    result = round(df.groupby('type')['Item_Price'].mean(),2)
    print(result,'\n')
    print('the average price of drinks is 8.42')
    print('the average price of food is 4.88')


except Exception as e:
    print(e)
    print('failed to read data')
    
