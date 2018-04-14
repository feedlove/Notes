# Load the necessary packages and modules
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
# Commodity Channel Index 
def CCI(data, ndays): 
	TP = (data['High'] + data['Low'] + data['Close']) / 3 
	CCI = pd.Series((TP - TP.rolling(window=ndays,center=False).mean().dropna()) / (0.015 * TP.rolling(window=ndays, center=False).std().dropna()),name = 'CCI') 
	data = data.join(CCI) 
	return data
# Retrieve the Nifty data from Yahoo finance:
data = pdr.get_data_yahoo('AAPL',start='1/1/2010', end='1/1/2016')
data = pd.DataFrame(data)
# Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
NIFTY_CCI = CCI(data,20)
CCI = NIFTY_CCI['CCI']
# Plotting the Price Series chart and the Commodity Channel index below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['Close'],lw=1)
plt.title('NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(CCI,'k',lw=0.75,linestyle='-',label='AAPL')
plt.legend(loc=2,prop={'size':9.5})
plt.ylabel('CCI values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()