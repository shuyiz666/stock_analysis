'''
assignment3: naive bayesian
question1: implement a Gaussian naive bayesian classifier and compute its accuracy for year 2
'''
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

wd = os.getcwd()
ticker = 'ZSAN'
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_label.csv')
df = pd.read_csv(ticker_file)

training = df[df['Year']==2017]
testing = df[df['Year']==2018]

df_1 =  training[training['label']=='green']
mu_mu_1 = df_1['mean_return'].values.mean()
mu_sigma_1 = df_1['mean_return'].values.std()
sigma_mu_1 = df_1['volatility'].values.mean()
sigma_sigma_1 = df_1['volatility'].values.std()

df_2 =  training[training['label']=='red']
mu_mu_2 = df_2['mean_return'].values.mean()
mu_sigma_2 = df_2['mean_return'].values.std()
sigma_mu_2 = df_2['volatility'].values.mean()
sigma_sigma_2 = df_2['volatility'].values.std()

p_green = len(df_1)/len(training)
p_red = len(df_2)/len(training)

prediction = []
for index, row in testing.iterrows():
    mu = row['mean_return']
    sigma = row['volatility']

    prob_mu_green = norm.pdf((mu-mu_mu_1/mu_sigma_1))
    prob_sigma_green = norm.pdf((sigma- sigma_mu_1)/sigma_sigma_1)

    prob_mu_red = norm.pdf((mu-mu_mu_2/mu_sigma_2))
    prob_sigma_red = norm.pdf((sigma- sigma_mu_2)/sigma_sigma_2)

    # unnormalized probabilities
    posterior_red = p_red *prob_mu_red *prob_sigma_red
    posterior_green = p_green *prob_mu_green *prob_sigma_green

    normalized_red = posterior_red /(posterior_red + posterior_green)
    normalized_green = posterior_green /(posterior_red + posterior_green)

    if normalized_red > normalized_green:
        prediction.append('red')
    else:
        prediction.append('green')

print('accuracy for year 2 is','%s%%'%round(sum(np.array(prediction)==testing[['label']].values.ravel())/len(testing[['label']].values.ravel())*100,2))

