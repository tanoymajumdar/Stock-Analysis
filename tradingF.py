import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

dfInput1_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitGC=FBitcoinInputtrain.csv')
dfInput2_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitSI=FBitcoinInputtrain.csv')
dfInput3_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitCL=FBitcoinInputtrain.csv')
dfInput4_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGit^GSPCBitcoinInputtrain.csv')
dfInput5_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGit^GSPCEthereumInputtrain.csv')
dfInput6_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitCL=FEthereumInputtrain.csv')
dfInput7_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitSI=FEthereumInputtrain.csv')
dfInput8_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitGC=FEthereumInputtrain.csv')

dfInput1_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitGC=FEthereumInputtest.csv')
dfInput2_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitSI=FEthereumInputtest.csv')
dfInput3_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitCL=FEthereumInputtest.csv')
dfInput4_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGit^GSPCEthereumInputtest.csv')
dfInput5_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitGC=FBitcoinInputtest.csv')
dfInput6_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitSI=FBitcoinInputtest.csv')
dfInput7_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitCL=FBitcoinInputtest.csv')
dfInput8_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGit^GSPCBitcoinInputtest.csv')

dfOutput1_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGit^GSPCBitcoinOutputtrain.csv')
dfOutput2_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitSI=FBitcoinOutputtrain.csv')
dfOutput3_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitGC=FBitcoinOutputtrain.csv')
dfOutput4_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitCL=FBitcoinOutputtrain.csv')
dfOutput5_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitSI=FEthereumOutputtrain.csv')
dfOutput6_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitGC=FEthereumOutputtrain.csv')
dfOutput7_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitCL=FEthereumOutputtrain.csv')
dfOutput8_train = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGit^GSPCEthereumOutputtrain.csv')

dfOutput1_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGit^GSPCEthereumOutputtest.csv')
dfOutput2_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitSI=FEthereumOutputtest.csv')
dfOutput3_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitGC=FEthereumOutputtest.csv')
dfOutput4_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitCL=FEthereumOutputtest.csv')
dfOutput5_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitSI=FBitcoinOutputtest.csv')
dfOutput6_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitGC=FBitcoinOutputtest.csv')
dfOutput7_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGitCL=FBitcoinOutputtest.csv')
dfOutput8_test = pd.read_csv(r'C:\Users\Tanoy Majumdar\Desktop\ArranoGit^GSPCBitcoinOutputtest.csv')

inputLayerTrainBTC = pd.concat([dfInput1_train, dfInput2_train, dfInput3_train, dfInput4_train ], axis=1, sort=False)
inputLayerTrainETH = pd.concat([dfInput5_train, dfInput6_train, dfInput7_train, dfInput8_train ], axis=1, sort=False)
inputLayerTestBTC = pd.concat([dfInput1_test, dfInput2_test, dfInput3_test, dfInput4_test], axis =1, sort=False)
inputLayerTestETH = pd.concat([dfInput5_test, dfInput6_test, dfInput7_test, dfInput8_test], axis =1, sort=False)
outputLayerTrainBTC = pd.concat([dfOutput1_train, dfOutput2_train, dfOutput3_train, dfOutput4_train ], axis = 1, sort = False)
outputLayerTrainETH = pd.concat([dfOutput5_train, dfOutput6_train, dfOutput7_train, dfOutput8_train ], axis = 1, sort = False)
outputLayerTestBTC = pd.concat([dfOutput1_test, dfOutput2_test, dfOutput3_test, dfOutput4_test ], axis = 1, sort = False)
outputLayerTestETH = pd.concat([dfOutput5_test, dfOutput6_test, dfOutput7_test, dfOutput8_test ], axis = 1, sort = False)

#inputLayerTrainBTC = inputLayerTrainBTC.drop(['change_cry', 'change_comm', 'correlations'], axis = 1)
#inputLayerTestBTC = inputLayerTestBTC.drop(['change_cry', 'change_comm', 'correlations'], axis = 1)
print(inputLayerTrainBTC)
print(outputLayerTrainBTC)


model = Sequential()
model.add(Dense(50, input_dim=inputLayerTrainBTC.shape[1], activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(inputLayerTrainBTC, outputLayerTrainBTC.iloc[:,0], epochs=50, batch_size=8)
_, accuracy = model.evaluate(inputLayerTrainBTC, outputLayerTrainBTC.iloc[:,0])
print('Accuracy: %.2f' % (accuracy*100))

closing_price = model.predict(inputLayerTestBTC)

print(closing_price)
print(outputLayerTestBTC.iloc[:,0])
