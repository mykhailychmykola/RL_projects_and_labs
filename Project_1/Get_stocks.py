import quandl
import numpy as np
import pickle


API_KEY = 'GR5ysvVM3HzDtjkPoPyi'
stock_data = {}

stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB", "JNJ", "XOM", "JPM", "WFC", "BAC", "GE", "T", "PG", "WMT", "V"]

for i in stocks:
    a = quandl.get('WIKI/' + i, start_date="2012-12-31", end_date="2017-04-30", api_key=API_KEY)
    stock_data[i] = np.log(a['Adj. Close'] / a['Adj. Open']).values


pickle.dump(stock_data, open("stock_data.p", "wb"))



