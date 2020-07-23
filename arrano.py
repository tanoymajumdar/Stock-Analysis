import csv 
import pandas as pd 
import sys
import seaborn as sb  
from sklearn.metrics import mean_squared_error
import numpy as np
from pandas.io.json import json_normalize
import json
import lxml.html as lh
from yahoofinancials import YahooFinancials
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
#import pandas_datareader as web
from mpl_finance import candlestick_ohlc
import san
import math
from scipy.stats.stats import pearsonr   


"""
API request of crypto from Santiment...
"""
def crypto_p(crypto, start, end):
    df = san.get('projects/all')
    ii = df.index[df['name'] == crypto]
    li = str(df['slug'][ii]).split()
    df1 = san.get("ohlcv/"+li[1], from_date=start, to_date=end, interval="1d")
    df1 = df1.assign(returns = df1['closePriceUsd'] - df1['openPriceUsd'])  #Returns Calculation per day
    return(df1)
    
"""
API call from YahooFinancials to get OHLCV data for commodity
"""
def yahoo_fina(commodity, start, end, interval):
    name = YahooFinancials(commodity)
    price = name.get_historical_price_data(start, end, 'daily')
    df = json_normalize(price[commodity], 'prices')
    df = df.assign(returns = df['close'] - df['open']) #Returns calculations per day
    df = df.set_index('formatted_date')
    idx = pd.date_range(start, end)
    df.index = pd.DatetimeIndex(df.index)

    df = df.reindex(idx, fill_value = None)

    return(df)

"""
Candlestick graphs for OHLC values of Crypto
"""  
def plotGraphs_crypto(crypto, start, end, interval):
    df_crypto = crypto_p(crypto, start, end)
    df_crypto = df_crypto.drop(['returns'], axis = 1)
    df_crypto = df_crypto.rename(columns={"openPriceUsd": "Open", "closePriceUsd": "Close", "highPriceUsd": "High", "lowPriceUsd":"Low"})
    df_crypto = df_crypto.drop(['volume', 'marketcap'], axis=1)
    df_crypto = df_crypto[[ 'Open', 'High', 'Low', 'Close']]
    df_crypto.reset_index(inplace=True)
    df_crypto['datetime'] = pd.to_datetime(df_crypto['datetime'])
    df_crypto['datetime'] = df_crypto['datetime'].apply(mdates.date2num)
    df_crypto = df_crypto.astype(float)
    fig, ax = plt.subplots()
    candlestick_ohlc(ax, df_crypto.values, width=0.6, colorup='blue', colordown='black', alpha=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    fig.suptitle('Daily Candlestick Chart of Bitcoin')
    date_format = mdates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    fig.tight_layout()
    plt.show()
    #return (plt)

"""
Candlestick graphs for OHLC values of Commodities
"""
def plotGraphs_commodity(commodity, start, end, interval):
    df_commodity = yahoo_fina(commodity, start, end, interval)
    df_commodity = df_commodity.drop(['returns'], axis=1)
    df_commodity = df_commodity.drop(['adjclose', 'date', 'volume'], axis = 1)
    df_commodity = df_commodity[['open', 'high', 'low', 'close']]
    df_commodity.reset_index(inplace=True)
    df_commodity['index'] = pd.to_datetime(df_commodity['index'])
    df_commodity['index'] = df_commodity['index'].apply(mdates.date2num)
    df_commodity = df_commodity.astype(float)
    fig, ax = plt.subplots()
    candlestick_ohlc(ax, df_commodity.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    fig.suptitle('Daily Candlestick Chart of' + commodity)
    date_format = mdates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    fig.tight_layout()
    #plt.show()
    return (plt)

"""
Function to help create a dataframe with returns of Crypto and returns of Commodity
"""
def returns_data(commodity, crypto, start, end, interval):
    df_commodity = yahoo_fina(commodity, start, end, interval)
    df_commodity.reset_index(inplace=True)
    df_crypto = crypto_p(crypto, start, end)
    df_crypto.reset_index(inplace=True)
    df = pd.DataFrame().assign(returns_crypto =  df_crypto['returns'] ,returns_com = df_commodity['returns'])
    return(df)

"""
Creates a graph by finding pearson correlations between returns of crypto and commodity
Weekly correlations.
"""
def returns_graph(commodity, crypto, start, end, interval, cor_type):
    df = returns_data(commodity, crypto, start, end, interval)
    corr_values = []
    if interval == 'weekly':
        for i in range(0, df.shape[0], 7):
            a = df.loc[i:i+7, :].corr(method = cor_type)
            corr_values.append(a['returns_crypto']['returns_com'])
    elif interval == 'monthly':
        for i in range(0, df.shape[0], 30):
            a = df.loc[i:i+30, :].corr(method = cor_type)
            corr_values.append(a['returns_crypto']['returns_com'])
    return (corr_values)

    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns correlation')
    ax.plot(corr_values)
    plt.show()


def price_correlation(commodity, crypto, start, end, interval):
    df_com = yahoo_fina(commodity, start, end, interval)
    df_com.reset_index(inplace = True)
    df2_cryp = crypto_p(crypto, start, end)
    df2_cryp.reset_index(inplace = True)
    df = pd.DataFrame().assign(price_crypto = df2_cryp['closePriceUsd'], price_commodity = df_com['close'])
    avg_price_com = (df['price_commodity'].sum())/df.shape[0]
    avg_price_crypto = (df['price_crypto'].sum())/df.shape[0]
    com_dev = pow(df['price_commodity'] - avg_price_com, 2)
    cry_dev = pow(df['price_crypto'] - avg_price_crypto, 2)
    df = df.assign(deviation = com_dev * cry_dev)
    com_dev_f = math.sqrt(com_dev.sum())
    cry_dev_f = math.sqrt(cry_dev.sum())
    sd = com_dev_f * cry_dev_f
    correlation = math.sqrt(df['deviation'].sum())/sd
    #return(correlation)
    return(df)

"""
Takes closed price of commodity and crypto and finds pearson correlation values
Weekly time frame.
"""
def correlations(commodity, crypto, start, end, interval):
    df = price_correlation(commodity, crypto, start, end, interval)
    df = df.drop(['deviation'], axis = 1)
    """ 
    
    """
    corr_values_week = []
    if interval == 'weekly':
        for i in range(0, df.shape[0], 9):
            a = df.loc[i:i+9, :].corr(method = 'pearson')
            corr_values_week.append(a['price_crypto']['price_commodity'])
    corr_values_biday = []

    dfCry = df.iloc[3:,0:1]
    dfCry.reset_index(inplace = True)
    dfCry = dfCry.drop(['index'], axis = 1)
    df = df.drop(df.index[-3:])
    df = df.drop(['price_crypto'], axis = 1)
    df = df.assign(price_crypto = dfCry)
    for i in range(0, df.shape[0], 3):
        a = df.loc[i:i+3, :].corr(method = 'pearson')
        corr_values_biday.append(a['price_crypto']['price_commodity'])

    """
    Fixing the last correlations for 3 day frame by removing an incomplete week if it exists.
    """
    if len(corr_values_biday)%3 !=0 or len(corr_values_biday)%len(corr_values_week)!=0:
        corr_values_week.pop()                                                                  #Incomplete 9 day frame is dropped.
        corr_values_biday = corr_values_biday[:len(corr_values_biday)-len(corr_values_biday)%3] #Dropping extra observations from incomplete 9 day frame.
    
    df9day = pd.DataFrame().assign(weekframe = corr_values_week)

    final_week = []
    """
    9 day correlations range adjusted with the 3 day correlations range by repeating entry into dataframe.
    """
    for i in range(0, len(corr_values_week)):
        for j in range (0,3):
            final_week.append(corr_values_week[i])

    correlations = pd.DataFrame().assign(weekly = final_week, trippledays = corr_values_biday)
    dfarray = []
    dfarray.append(correlations)
    dfarray.append(df9day)
    return(dfarray)
    

    """
    elif interval == 'monthly':
        for i in range(0, df.shape[0], 30):
            a = df.loc[i:i+30, :].corr(method = cor_type)
            corr_values.append(a['price_crypto']['price_commodity'])
    
    #return(corr_values)

    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Price correlation')
    ax.plot(final_week)
    plt.show()
    """
    

"""
def main(argv):
    commodity = str(sys.argv[1])
    crypto = str(sys.argv[2])
    start = str(sys.argv[3])
    end = str(sys.argv[4])
    interval = str(sys.argv[5])
    #cor_type = str(sys.argv[6])
    #plotGraphs_crypto(crypto, start, end, interval)
    #plotGraphs_commodity(commodity, start, end, interval)
    correlations(commodity, crypto, start, end, interval)

if __name__ == "__main__":
   main(sys.argv[1:])    
"""
