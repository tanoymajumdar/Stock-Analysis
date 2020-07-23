import arrano_new
import arrano
import trader
import sys
import pandas as pd
import matplotlib.pyplot as plt

"""
Calculating average price of crypto for a 9 day window.
"""
def avgCrypto(commodity, crypto, start, end, interval):
    df1 = arrano.crypto_p(crypto, start, end)
    avg = []
    for i in range(0, df1.shape[0], 9):
        avg.append(df1.iloc[i:i+9]['closePriceUsd'].sum()/9)
    df = pd.DataFrame().assign(crypto = avg)
    return(df)

"""
This function was for analysis purpose only.
"""
def execute(commodity, crypto, start, end, interval):
    array = arrano_new.findPattern(commodity, crypto, start, end, interval)
    df1 = arrano.crypto_p(crypto, start, end)
    df2 = arrano.yahoo_fina(commodity, start, end, interval)
    cryArrayW = []
    comArrayW = []
    for i in array:
        cryArrayW.append((df1.iloc[i*9 +8]['closePriceUsd']-df1.iloc[i*9]['closePriceUsd'])/df1.iloc[i*9 +8]['closePriceUsd'])
        comArrayW.append((df2.iloc[i*9 +8]['close']-df2.iloc[i*9]['close'])/df2.iloc[i*9 +8]['close'])
    cryArrayT = []
    comArrayT = []
    for i in array:
        cryArrayT.append((df1.iloc[i*9 +2]['closePriceUsd']-df1.iloc[i*9]['closePriceUsd'])/df1.iloc[i*9 +2]['closePriceUsd'])
        comArrayT.append((df2.iloc[i*9 +2]['close']-df2.iloc[i*9]['close'])/df2.iloc[i*9 +2]['close'])
    
        
    df = pd.DataFrame().assign(cryW = cryArrayW, comW = comArrayW, cryT = cryArrayT, comT = comArrayT)
    
    print(df)
"""
Final algorithm for predicting movement.
"""
def final(commodity, crypto, start, end, interval):
    skips = arrano_new.findPattern(commodity, crypto, start, end, interval) #Returns the average skips before prices highly correlated.
    skips = round(skips)
    dfData = arrano.correlations(commodity, crypto, start, end, interval)[0] # Gives 9 day correlation and 3 day correlations
    df1 = arrano.crypto_p(crypto, start, end) # Rteurns a dataframe of OHLC values of crypto
    df2 = arrano.yahoo_fina(commodity, start, end, interval) # Rteurns a dataframe of OHLC values of crypto
    df3 = avgCrypto(commodity, crypto, start, end, interval) # Rteurns a dataframe of average prices of crypto for a 9 day range
    df4 = trader.getBestTrades(commodity, crypto, start, end, interval) # Returns a data frame of High and Low prices of crypto for a 9 day range.
    cryArrayT = []
    comArrayT = []
    index_date = []
    """
    Running a loop on the dataset using the average skips to find trading weeks.
    """
    for i in range(9, df1.shape[0] - df1.shape[0]%(9*skips), skips*9): #Running loop till values divisible by 9 and skips.
        cryArrayT.append((df1.iloc[i +2]['closePriceUsd']-df1.iloc[i]['closePriceUsd'])/df1.iloc[i+2]['closePriceUsd'])
        comArrayT.append((df2.iloc[i +2]['close']-df2.iloc[i]['close'])/df2.iloc[i +2]['close'])
    
    df = pd.DataFrame().assign(cryT = cryArrayT, comT = comArrayT, tripplesCorr = dfData[['trippledays']])
    semiPredict = []
    
    """
    Finding the change percentage by multiplying correlation value and change in price in the 3 day time frame.
    """
    for i in range(0, df.shape[0]):
        if df.iloc[i]['tripplesCorr'] > 0 and df.iloc[i]['cryT']>0:
            trade = df.iloc[i]['cryT']*df.iloc[i]['tripplesCorr']
            semiPredict.append(trade)
            index_date.append(i)
    print(len(semiPredict))
    
    final = []
    c = 0
    """
    Using the index of trading week, find the previous week average and predict deviated prices based on the calculated changes.
    """
    for i in index_date:
        trades = (df3.iloc[(i-1)*skips]['crypto']*semiPredict[c]) + df3.iloc[(i-1)*skips]['crypto']
        final.append(trades)
        c = c+1
    

    lowCry = []
    highCry = []

    """
    Getting trading week high and low prices to understand whether prediction lies in range.
    """
    for i in index_date:
        lowCry.append(df4.iloc[i*skips]['lowCrypto'])
        highCry.append(df4.iloc[i*skips]['highCrypto'])
    
    dataFrame = pd.DataFrame().assign(tradesFinal = final, lowWeek = lowCry, highWeek = highCry)
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Prices')
    ax.plot(dataFrame['lowWeek'], label = 'low')
    ax.plot(dataFrame['highWeek'], label = 'high')
    ax.plot(dataFrame['tradesFinal'], label = 'predicted')
    plt.legend(loc="upper left")
    plt.show()
    
"""
Main function to be executed using the command line.
"""
def main(argv):
    final(str(sys.argv[1]), str(sys.argv[2]),str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))

if __name__ == "__main__":
   main(sys.argv[1:])