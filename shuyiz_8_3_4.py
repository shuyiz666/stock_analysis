'''
assignment3: naive bayesian
question4: implement a trading strategy based on your labels for year 2 and compare the performance with the ”buy-and-hold” strategy. Which strategy results in a larger amount at the end of the year?
'''
import os
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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

# trading by predict_labels
money = 100
# flag = 0 only have money 1 only have stock
flag = 0
portfolio = 100
portfolios,portfolios_buy_hold = [],[]
i = 0
for index, row in testing.iterrows():
    # buy hold
    if i == 0:
        portfolios_buy_hold.append(100)
        stock_buy_hold = 100/row['Adj Close']
    else:
        portfolios_buy_hold.append(stock_buy_hold*row['Adj Close'])

    # trading with labels
    # red to green, buy stock
    if prediction[i] == 'green' and flag == 0:
        shares = money / row['Adj Close']
        money = 0
        flag = 1
        portfolio = shares * row['Adj Close']
    # green to green, do nothing
    elif prediction[i] == 'green' and flag == 1:
        portfolio = shares * row['Adj Close']
    # red to red, do nothing
    elif prediction[i] == 'red' and flag == 0:
        pass
    # green to red, sell stock
    elif prediction[i] == 'red' and flag == 1:
        money = shares * row['Adj Close']
        shares = 0
        flag = 0
        portfolio = money
    i += 1
    portfolios.append(portfolio)

if portfolios[-1] > portfolios_buy_hold[-1]:
    print('trading strategy based on naive bayesian predicted labels results in a larger amount at the end of the year')
elif portfolios[-1] == portfolios_buy_hold[-1]:
    print('trading strategy based on naive bayesian predicted labels results the same amount with buy hold strategy at the end of the year')
else:
    print('buy-and-hold results in a larger amount at the end of the year')

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.title('two strategies portfolio in 2018')
plt.xlabel('week_number', fontsize=14)
xlabels = testing['Week_Number']
plt.ylabel('portfolio', fontsize=14)
plt.yticks()
plt.plot(xlabels,portfolios)
plt.plot(xlabels,portfolios_buy_hold)
plt.legend(['trading with labels (naive bayesian)','buy hold'])
plt.show()

