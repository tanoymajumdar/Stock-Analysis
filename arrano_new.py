import san
import arrano
import sys
import matplotlib.pyplot as plt
import pandas as pd

from yahoofinancials import YahooFinancials
import json
from pandas.io.json import json_normalize

def yahoo_fina(commodity, start, end):
    name = YahooFinancials(commodity)
    price = name.get_historical_price_data(start, end, 'daily')
    df = json_normalize(price[commodity], 'prices')
    df = df.assign(returns = df['close'] - df['open'])
    df = df.set_index('formatted_date')
    idx = pd.date_range(start, end)
    df.index = pd.DatetimeIndex(df.index)

    df = df.reindex(idx, fill_value = None)

    return(df)

def moving_avg(commodity, start, end):
    df = yahoo_fina(commodity, start, end)
    data = df['close']
    df['MA'] = data.rolling(window=3).mean()
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Price correlation')
    ax.plot(df['returns'])
    ax.plot(df['close'])
    plt.show()

def findPattern(commodity, crypto, start, end, interval):
    dfweekly = arrano.correlations(commodity, crypto, start, end, interval)[1]
    dfData = arrano.correlations(commodity, crypto, start, end, interval)[0]
    count = 0
    avg_skips = []
    index_high = []
    for i in range(0,dfweekly.shape[0]-1):
        if dfweekly.iloc[i]['weekframe'] >= 0.5:
            while dfweekly.iloc[i+count+1]['weekframe'] <= 0.5 and i+count+1<dfweekly.shape[0]-1:   #Loop runs till the final entry only
                count = count +1
            avg_skips.append(count)
            index_high.append(i)
        count = 0
    trippledays = []
    weekframe = []
    for i in index_high:
        trippledays.append(dfData.iloc[i*3]['trippledays'])
        weekframe.append(dfweekly.iloc[i]['weekframe'])
    df = pd.DataFrame().assign(tripple = trippledays, week = weekframe) 
    array =[]
    for i in range(0, df.shape[0]):
        if df.iloc[i]['tripple'] == 1:
            array.append(df.iloc[i]['week'])
    
    #print(df)
    #print (sum(array)/len(array))   
    #print(sum(avg_skips)/len(avg_skips))
    #print(weekframe)
    return(sum(avg_skips)/len(avg_skips))


"""
def main(argv):
    
    #yahoo_fina('GC=F', str(sys.argv[1]), str(sys.argv[2]))
    #yahoo_fina('SI=F', str(sys.argv[1]), str(sys.argv[2]))
    #yahoo_fina('CL=F', str(sys.argv[1]), str(sys.argv[2]))
    #moving_avg('SI=F', str(sys.argv[1]), str(sys.argv[2]))
    findPattern(str(sys.argv[1]), str(sys.argv[2]),str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))

if __name__ == "__main__":
   main(sys.argv[1:]) 
"""