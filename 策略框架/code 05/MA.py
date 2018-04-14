# Moving Averages Code

# Load the necessary packages and modules
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt

# Simple Moving Average 
def SMA(data, ndays): 
	SMA = pd.Series(data['Close'].rolling(window=ndays).mean(), name = 'SMA') 
	data = data.join(SMA) 
	return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
	EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), name = 'EWMA_' + str(ndays)) 
	data = data.join(EMA) 
	return data

# Retrieve the Nifty data from Yahoo finance:
data = pdr.get_data_yahoo('AAPL',start='1/1/2010', end='1/1/2016')
data = pd.DataFrame(data) 
close = data['Close']

# Compute the 50-day SMA for NIFTY
n = 50
SMA_NIFTY = SMA(data,n)
SMA_NIFTY = SMA_NIFTY.dropna()
SMA = SMA_NIFTY['SMA']

# Compute the 200-day EWMA for NIFTY
ew = 200
EWMA_NIFTY = EWMA(data,ew)
EWMA_NIFTY = EWMA_NIFTY.dropna()
EWMA = EWMA_NIFTY['EWMA_200']

# Plotting the NIFTY Price Series chart and Moving Averages below
plt.figure(figsize=(9,5))
plt.plot(data['Close'],lw=1, label='NSE Prices')
plt.plot(SMA,'g',lw=1, label='50-day SMA (green)')
plt.plot(EWMA,'r', lw=1, label='200-day EWMA (red)')
plt.legend(loc=2,prop={'size':11})
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()