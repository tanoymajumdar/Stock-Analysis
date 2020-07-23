import arrano
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

def avgPrices(commodity, crypto, start, end, interval):
    df = arrano.price_correlation(commodity, crypto, start, end, interval)
    data = pd.DataFrame()
    to_drop_last = 0
    avg_crypto = []
    avg_comm = []
    ch_cry = []
    ch_com = []

    if df.shape[0]%8 != 0:
        to_drop_last = df.shape[0]%8 

    
    for i in range(0, df.shape[0]-to_drop_last, 2):
        a1 = (df.loc[i:i+2, ['price_crypto']].sum())/2
        avg_crypto.append(a1)
        a2 = (df.iloc[i+2]['price_crypto'] - df.iloc[i]['price_crypto'])/df.iloc[i+2]['price_crypto']
        ch_cry.append(a2)

    for i in range(0, df.shape[0]-to_drop_last, 2):
        a1 = (df.loc[i:i+2, ['price_commodity']].sum())/2
        avg_comm.append(a1)
        a2 = (df.iloc[i+2]['price_commodity'] - df.iloc[i]['price_commodity'])/df.iloc[i+2]['price_commodity']
        ch_com.append(a2)
    avg_crypto = [i.to_numpy()[0] for i in avg_crypto]
    avg_comm = [i.to_numpy()[0] for i in avg_comm]
    print(len(avg_crypto))
    print(len(avg_comm))
    data = data.assign(avg_cry = avg_crypto, avg_commodity = avg_comm, change_cry = ch_cry, change_comm = ch_com) 
    return(data)
"""
    for i in range (0, df.shape[0] - to_drop_last, 8):
        last = i+8
        first = i
        if pd.isna(df.iloc[i+8]['price_commodity']):
            j = i
            while pd.isna(df.iloc[j]['price_commodity']):
                j = j-1
            last = j
        if pd.isna(df.iloc[i]['price_commodity']):
            j = i
            while pd.isna(df.iloc[j]['price_commodity']):
                j = j+1
            first = j 
        b1 = (df.loc[first:last, ['price_commodity']].sum())/(last-first)
        avg_comm.append(b1)
        b2 = (df.iloc[last]['price_commodity'] - df.iloc[first]['price_commodity'])/df.iloc[last]['price_commodity']
        ch_com.append(b2)
"""
    

def helper(commodity, crypto, start, end, interval):
    df = avgPrices(commodity, crypto, start, end, interval)
    change_from_previous_comm = []
    change_from_previous_cry = []
    for i in range(0, df.shape[0]-1):
        if df.iloc[i]['change_comm']>0 and df.iloc[i+1]['change_comm']>df.iloc[i]['change_comm']:
            change_from_previous_comm.append(1)
        elif df.iloc[i]['change_comm']>0 and df.iloc[i+1]['change_comm']>0 and df.iloc[i]['change_comm']>df.iloc[i+1]['change_comm'] :
            change_from_previous_comm.append(0)
        elif df.iloc[i]['change_comm']>0 and df.iloc[i+1]['change_comm']<0:
            change_from_previous_comm.append(-1)
        elif df.iloc[i]['change_comm']<0 and df.iloc[i+1]['change_comm']<df.iloc[i]['change_comm']:
            change_from_previous_comm.append(1)
        elif df.iloc[i]['change_comm']<0 and df.iloc[i+1]['change_comm']<0 and df.iloc[i+1]['change_comm']>df.iloc[i]['change_comm'] :
            change_from_previous_comm.append(0)
        elif df.iloc[i]['change_comm']<0 and df.iloc[i+1]['change_comm']>0:
            change_from_previous_comm.append(-1)
        else:
            change_from_previous_comm.append(0)

        if df.iloc[i]['change_cry']>0 and df.iloc[i+1]['change_cry']>df.iloc[i]['change_cry']:
            change_from_previous_cry.append(1)
        elif df.iloc[i]['change_cry']>0 and df.iloc[i+1]['change_cry']>0 and df.iloc[i]['change_cry']>df.iloc[i+1]['change_cry'] :
            change_from_previous_cry.append(0)
        elif df.iloc[i]['change_cry']>0 and df.iloc[i+1]['change_cry']<0:
            change_from_previous_cry.append(-1)
        elif df.iloc[i]['change_cry']<0 and df.iloc[i+1]['change_cry']<df.iloc[i]['change_cry']:
            change_from_previous_cry.append(1)
        elif df.iloc[i]['change_cry']<0 and df.iloc[i+1]['change_cry']<0 and df.iloc[i]['change_cry']<df.iloc[i+1]['change_cry'] :
            change_from_previous_cry.append(0)
        elif df.iloc[i]['change_cry']<0 and df.iloc[i+1]['change_cry']>0:
            change_from_previous_cry.append(-1)
        else:
            change_from_previous_cry.append(0)
    #corr = arrano.correlations(commodity, crypto, start, end, interval, 'pearson')

    # If incomplete week remove corr value.
    # 1 extra value for change hence remove one corr always.
    #if len(corr) != df.shape[0]:
        #corr.pop()
    #corr.pop()
    datalist = pd.DataFrame().assign(future_ch_comm = change_from_previous_comm, future_ch_cry = change_from_previous_cry)
    return(datalist)

def correlations_analysis(commodity, crypto, start, end, interval):
    corr = arrano.correlations(commodity, crypto, start, end, interval)
    positive_relation_consecutive = []
    negative_relation_consecutive = []
    relations_change_inverse = []
    no_imp_change = []
    for i in range(0,len(corr)-1):
        if corr[i]>0.5:
            c = 1
            while corr[i+c]>0.5:
                c = c +1
            positive_relation_consecutive.append(c)
            c = 1
        elif corr[i]<-0.5:
            c = 1
            while corr[i+c]<-0.5:
                c = c+1
            negative_relation_consecutive.append(c)
            c = 1
    print(sum(positive_relation_consecutive)/len(corr))
    print(negative_relation_consecutive)
    print(no_imp_change)
    print(len(corr))



def getBestTrades(commodity, crypto, start, end, interval):
    to_drop_last = 0
    df1 = arrano.yahoo_fina(commodity, start, end, interval)
    if df1.shape[0]%7 != 0:
        to_drop_last = df1.shape[0]%7
    df1.reset_index(inplace = True)
    low_comm = []
    high_comm = []
    high_cry = []
    low_cry = []
    df2 = arrano.crypto_p(crypto, start, end)
    df2.reset_index(inplace = True)

    for i in range(9, df2.shape[0], 9):
        a1 = df2.loc[i:i+9, 'highPriceUsd'].max()
        high_cry.append(a1)
        a2 = df2.loc[i:i+9, 'lowPriceUsd'].max()
        low_cry.append(a2)

    for i in range (0, df1.shape[0] - to_drop_last, 7):
        last = i+7
        first = i
        if pd.isna(df1.iloc[i+7]['high']):
            j = i
            while pd.isna(df1.iloc[j]['high']):
                j = j-1
            last = j
        if pd.isna(df1.iloc[i]['high']):
            j = i
            while pd.isna(df1.iloc[j]['high']):
                j = j+1
            first = j 
        b1 = df1.loc[first:last, 'high'].max()
        high_comm.append(b1)
        b2 = df1.loc[first:last, 'low'].min()
        low_comm.append(b2)
    data = pd.DataFrame().assign(highCrypto = high_cry, lowCrypto = low_cry)
    return(data)
    """
    data_price = avgPrices(commodity, crypto, start, end, interval)

    trade_price_crypto = []
    trade_price_commodity = []
    
    trade_price_crypto.append(-(data.iloc[0]['lowCrypto']))
    trade_price_commodity.append(-(data.iloc[0]['lowComm']))

    for i in range (1, data.shape[0]):
        if data_price.iloc[i]['change_cry']>0:
            trade_price_crypto.append(data.iloc[i]['highCrypto'])
        elif data_price.iloc[i]['change_cry']<0:
            trade_price_crypto.append(-(data.iloc[i]['lowCrypto']))
        
        if data_price.iloc[i]['change_comm']>0:
            trade_price_commodity.append(data.iloc[i]['highComm'])
        elif data_price.iloc[i]['change_comm']<0:
            trade_price_commodity.append(-(data.iloc[i]['lowComm']))
        else:
            trade_price_commodity.append(np.nan)
    finals = pd.DataFrame().assign(trade_crpto = trade_price_crypto, trade_comm = trade_price_commodity)
    return(finals)
    """

def MakeInputReady(commodity, crypto, start, end, interval):
    df1 = avgPrices(commodity, crypto, start, end, interval)
    df1 = df1.drop(['avg_commodity', 'avg_cry'], axis = 1)
    df2 = helper(commodity, crypto, start, end, interval)
    df1.drop(df1.tail(1).index,inplace=True)
    inputLayer = pd.DataFrame()
    inputLayer = pd.concat([df1, df2], axis=1, sort=False)
    return(inputLayer)

def learningModels(commodity, crypto, start, end, interval):
    dfInput = MakeInputReady(commodity, crypto, start, end, interval)
    #dfOutput = getBestTrades(commodity, crypto, start, end, interval)
    #dfOutput = dfOutput.drop(['trade_comm'], axis = 1)
    #dfOutput.drop(dfOutput.tail(1).index,inplace=True)
    
    X_train = dfInput.iloc[0:round(dfInput.shape[0]*0.9), 0:4]
    X_test = dfInput.iloc[-round(dfInput.shape[0]*0.1): , 0:4]

    y_train = dfInput.iloc[0:round(dfInput.shape[0]*0.9),-1:]
    y_test = dfInput.iloc[-round(dfInput.shape[0]*0.1):,-1:]

    X_train.to_csv('C:\\Users\\Tanoy Majumdar\\Desktop\\ArranoGit'+commodity+crypto+'Inputtrain.csv', index=False)
    X_test.to_csv('C:\\Users\\Tanoy Majumdar\\Desktop\\ArranoGit'+commodity+crypto+'Inputtest.csv', index=False)

    y_train.to_csv('C:\\Users\\Tanoy Majumdar\\Desktop\\ArranoGit'+commodity+crypto+'Outputtrain.csv', index=False)
    y_test.to_csv('C:\\Users\\Tanoy Majumdar\\Desktop\\ArranoGit'+commodity+crypto+'Outputtest.csv', index=False)

def inputting(commodity, crypto, start, end, interval):
    dfCorr = arrano.correlations(commodity, crypto, start, end, interval)
    dfChanges = helper(commodity, crypto, start, end, interval)
    dfCorr = dfCorr.drop(dfCorr.index[-1:])
    dfavgPrice_change = avgPrices(commodity, crypto, start, end, interval)
    dfavgPrice_change = dfavgPrice_change.drop(dfavgPrice_change.index[-1:])
    dfHighLow = getBestTrades(commodity, crypto, start, end, interval)
    dfHighLow = dfHighLow.drop(dfHighLow.index[-1:])
    data = pd.concat([dfCorr, dfChanges, dfavgPrice_change, dfHighLow], axis = 1)
    print(data)
    indexes = []
    data = data[(data['weekly']>=0.5) | (data['weekly']<=-0.5)]
    data = data[(data['biweekly']>=0.5) | (data['biweekly']<=-0.5)]
    positive_to_positive = []
    positive_to_negative = []
    negative_to_positive = []
    negative_to_negative = []
    for index, row in data.iterrows():
        if 0.5<row['weekly'] and 0.5<row['biweekly']:
            positive_to_positive.append(index)
        elif row['weekly']<-0.5 and row['biweekly']<-0.5:
            positive_to_negative.append(index)
        elif row['weekly']<-0.5 and row['biweekly']>0.5:
            negative_to_positive.append(index)
        else:
            negative_to_negative.append(index)
            #print(data.iloc[i]['weekly'])

    print(negative_to_negative)
    print(negative_to_positive)
    print(positive_to_negative)
    print(positive_to_positive)

def dualdayCorr(commodity, crypto, start, end, interval):
    df1 = avgPrices(commodity, crypto, start, end, interval)
    df2 = arrano.correlations(commodity, crypto, start, end, interval)
    data = pd.concat([df1, df2], axis = 1, sort = False)
    print(data)
"""
    model = Sequential()
    model.add(Dense(50, input_dim=X_train.shape[1], activation='sigmoid'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=8)

    closing_price = model.predict(X_test)

    print(closing_price)

"""

def main(argv):
    #avgPrices(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))
    #correlating(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))
    #helper(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))
    #getBestTrades(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))
    #MakeInputReady(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))
    #learningModels(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))
    #correlations_analysis(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))
    #inputting(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))
    dualdayCorr(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5]))

if __name__ == "__main__":
   main(sys.argv[1:]) 

