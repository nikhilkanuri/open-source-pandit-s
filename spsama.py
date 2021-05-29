from json import load
from altair.vegalite.v4.schema.core import Data
from matplotlib.pyplot import axis, close, title
from nltk.tree import Tree
from nltk.util import pr
import yfinance as yf
import streamlit as st
from plotly import graph_objs as go
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from stockstats import StockDataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime

st.title('STOCK PREDICTION WITH SENTIMENT ANALYSIS AND EMA')
dictoin = {"AMAZON.COM, Inc." :"AMZN","TESLA, Inc." : "TSLA","GOOGLE" : "GOOG","MICROSOFT" : "MSFT", "APPLE Inc." : "AAPL", "ABBOTT LABORATORIES" : "ABT"}
stocks= ("AMAZON.COM, Inc.", "TESLA, Inc.", "GOOGLE", "MICROSOFT", "APPLE Inc.", "ABBOTT LABORATORIES")

st.sidebar.header("Settings")
st.sidebar.subheader("Select Company")
selected_ticker_company_name = st.sidebar.selectbox('',stocks)
selected_ticker = dictoin[selected_ticker_company_name]

days = st.slider('Select number of days:', 5, 100)
period = str(days)+"d"

st.sidebar.subheader("Stocks Trend Plot")
check1 = st.sidebar.checkbox("Open")
check3 = st.sidebar.checkbox("High")
check4 = st.sidebar.checkbox("Low")
check5 = st.sidebar.checkbox("Volume")

st.sidebar.subheader("Predicted Stocks Trend Plot")
predicted_open_check = st.sidebar.checkbox("OPEN")
predicted_high_check = st.sidebar.checkbox("HIGH")
predicted_low_check = st.sidebar.checkbox("LOW")
predicted_volume_check = st.sidebar.checkbox("VOLUME")

st.sidebar.subheader("Scored mean Plot")
compound_check = st.sidebar.checkbox("Compound")
negative_check = st.sidebar.checkbox("Negative")
neutral_check = st.sidebar.checkbox("Neutral")
positive_check = st.sidebar.checkbox("Positive")

st.sidebar.subheader("Technical Indicators")
sma_check = st.sidebar.checkbox("Simple Moving Average")
ema_check = st.sidebar.checkbox("Exponential Moving Average")
macd_check = st.sidebar.checkbox("Moving Average Convergence/Divergence")

@st.cache
def load_data(ticker, period):
    data = yf.Ticker(ticker)
    print(data.info)
    df = data.history(period=period)
    rdf = df.iloc[::-1]
    print(df)
    print(rdf)
    return rdf

data_load_state = st.text("Loading Data")
stock_data = load_data(selected_ticker, period)

data_load_state.text("Loading Data...Done!")
st.subheader('Stocks Trend')
stock_data = stock_data.reset_index()
stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
print(type(stock_data))
print(stock_data)
st.write(stock_data)

old = stock_data.reset_index()
for i in ['Open', 'High', 'Close', 'Low', 'Volume']: 
      old[i]  =  old[i].astype('float64')

def plot_close():
    fig = go.Figure()
    st.subheader("Closing Trend")
    fig.add_trace(go.Scatter(x=old['Date'], y=old['Close'], name='stock_close'))
    fig.layout.update(xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
plot_close()


if check1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=old['Date'], y=old['Open'], name='stock_open'))
    fig.layout.update(title_text = 'Open', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

if check3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=old['Date'], y=old['High'], name='stock_high'))
    fig.layout.update(title_text = 'High', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
if check4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=old['Date'], y=old['Low'], name='stock_low'))
    fig.layout.update(title_text = 'Low', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
if check5:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=old['Date'], y=old['Volume'], name='stock_volume'))
    fig.layout.update(title_text = 'Volume', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)


sd1 = pd.DataFrame(stock_data['Open'])
numpy_array1 = sd1.to_numpy()
np.savetxt("open.txt", numpy_array1, fmt = "%d")

sd2 = pd.DataFrame(stock_data['High'])
numpy_array2 = sd2.to_numpy()
np.savetxt("high.txt", numpy_array2, fmt = "%d")

sd3 = pd.DataFrame(stock_data['Low'])
numpy_array3 = sd3.to_numpy()
np.savetxt("low.txt", numpy_array3, fmt = "%d")

sd4 = pd.DataFrame(stock_data['Close'])
numpy_array4 = sd4.to_numpy()
np.savetxt("close.txt", numpy_array4, fmt = "%d")

sd5 = pd.DataFrame(stock_data['Volume'])
numpy_array5 = sd5.to_numpy()
np.savetxt("volume.txt", numpy_array5, fmt = "%d")


X = []
for i in range(days):
    print(i)
    X.append(i+1)
per = days+1
y = np.loadtxt('open.txt', dtype=int)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
x = [[per]]
y_pred = regressor.predict(x)
print(x,"   ;  ",y_pred)
date = datetime.date.today() + datetime.timedelta(days=1)
dfe = pd.DataFrame(y_pred)
dfe.insert(0,"Date", date, True)
dfe.columns = ["Date", "Open"]

y = np.loadtxt('high.txt', dtype=int)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(x)
print(x,"   ;  ",y_pred)
dfe.insert(2, "High", y_pred, True)

y = np.loadtxt('low.txt', dtype=int)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(x)
print(x,"   ;  ",y_pred)
dfe.insert(3, "Low", y_pred, True)

y = np.loadtxt('close.txt', dtype=int)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(x)
print(x,"   ;  ",y_pred)
dfe.insert(4, "Close", y_pred, True)

y = np.loadtxt('volume.txt', dtype=int)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(x)
print(x,"   ;  ",y_pred)
dfe.insert(5, "Volume", y_pred, True)

dfe.insert(6, "Dividends", "0", True)
dfe.insert(7, "Stock Splits", "0", True)

st.subheader('Predicted Stocks Trend')
st.write(dfe)

dfe1=dfe.append(stock_data, ignore_index = True)
st.subheader('Stocks Trend with Predicted Trend')
st.write(dfe1)

# old = stock_data.reset_index()
# for i in ['Open', 'High', 'Close', 'Low', 'Volume']: 
#       old[i]  =  old[i].astype('float64')

def plot_close():
    fig = go.Figure()
    st.subheader("Closing Trend")
    fig.add_trace(go.Scatter(x=dfe1['Date'], y=dfe1['Close'], name='predicted_stock_close'))
    fig.layout.update(xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
plot_close()


if predicted_open_check:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfe1['Date'], y=dfe1['Open'], name='predicted_stock_open'))
    fig.layout.update(title_text = 'Open', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

if predicted_high_check:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfe1['Date'], y=dfe1['High'], name='predicted_stock_high'))
    fig.layout.update(title_text = 'High', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

if predicted_low_check:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfe1['Date'], y=dfe1['Low'], name='predicted_stock_low'))
    fig.layout.update(title_text = 'Low', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

if predicted_volume_check:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfe1['Date'], y=dfe1['Volume'], name='predicted_stock_volume'))
    fig.layout.update(title_text = 'Volume', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)


@st.cache
def news(selected_ticker):
    finwiz_url = "https://finviz.com/quote.ashx?t="
    news_tables = {}
    news_ticker = [selected_ticker]
    for ticker in news_ticker:
        url = finwiz_url + ticker
        req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
        response = urlopen(req)    
        html = BeautifulSoup(response)
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table
    news_ticker_web_data = news_tables[selected_ticker]
    news_tr = news_ticker_web_data.findAll('tr')

    for i, table_row in enumerate(news_tr):
        a_text = table_row.a.text
        td_text =  table_row.td.text
        print(a_text)
        print(td_text)
        if i == 3:
            break

    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()
            if len(date_scrape) == 1:
                time = date_scrape[0]
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            ticker = file_name.split('_')[0]
            parsed_news.append([ticker, date, time, text])
    return parsed_news

parsed_news = news(selected_ticker)
st.subheader('News')
news_df = pd.DataFrame(parsed_news)
st.write(news_df)
print(parsed_news)

nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()
columns = ['ticker', 'date', 'time', 'headline']
parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)
scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
scores_df = pd.DataFrame(scores)
parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
print(parsed_and_scored_news)
st.subheader('Scored News')
st.write(parsed_and_scored_news)

mean_scores = parsed_and_scored_news.groupby(['ticker','date']).mean()
mean_scores = mean_scores.unstack()
print("Thisismean", mean_scores)

mean_scores_compound =mean_scores.xs('compound', axis="columns").transpose()
mean_scores_compound = mean_scores_compound.reset_index()
mean_scores_compound.columns = ['date','compound']

mean_scores_neg = mean_scores.xs('neg', axis='columns').transpose()
mean_scores_neg = mean_scores_neg.reset_index()
mean_scores_neg.columns = ['date','negative']

mean_scores_neu = mean_scores.xs('neu', axis='columns').transpose()
mean_scores_neu = mean_scores_neu.reset_index()
mean_scores_neu.columns = ['date','   neutral']

mean_scores_pos = mean_scores.xs('pos', axis='columns').transpose()
mean_scores_pos = mean_scores_pos.reset_index()
mean_scores_pos.columns = ['date','positive']


smcol1, smcol2 = st.beta_columns(2)
smcol3, smcol4 = st.beta_columns(2)
with smcol1:
    st.subheader("Compound")
    st.write(mean_scores_compound[::-1])
with smcol2:
    st.subheader("Negative")
    st.write(mean_scores_neg[::-1])
with smcol3:
    st.subheader("Neutral")
    st.write(mean_scores_neu[::-1])
with smcol4:
    st.subheader("Positive")
    st.write(mean_scores_pos[::-1])

if compound_check:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_scores_compound['date'], y=mean_scores_compound['compound'], name='scored_mean_compound_plot'))
    fig.layout.update(title_text = 'Compound', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

if negative_check:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_scores_neg['date'], y=mean_scores_neg['negative'], name='scored_mean_negative_plot'))
    fig.layout.update(title_text = 'Negative', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
if neutral_check:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_scores_neu['date'], y=mean_scores_neu['   neutral'], name='scored_mean_neutral_plot'))
    fig.layout.update(title_text = 'Neutral', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
if positive_check:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_scores_pos['date'], y=mean_scores_pos['positive'], name='scored_mean_positive_plot'))
    fig.layout.update(title_text = 'Positive', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

@st.cache
def sma(stock_data):
    stocks = StockDataFrame.retype(stock_data[["Open", "Close", "High", "Low", "Volume"]])
    sma =  stocks[f"close_{days}_sma"]
    sma = sma.reset_index()
    sma.columns = ["Date", "SMA"]
    return sma

@st.cache
def ema(stock_data):
    stocks = StockDataFrame.retype(stock_data[["Open", "Close", "High", "Low", "Volume"]])
    ema =  stocks[f"close_{days}_ema"]
    ema = ema.reset_index()
    ema.columns = ["Date", "EMA"]
    return ema

@st.cache
def macd(stock_data):
    stocks = StockDataFrame.retype(stock_data[["Open", "Close", "High", "Low", "Volume"]])
    macd =  stocks["macd"]
    macd = macd.reset_index()
    macd.columns = ["Date", "MACD"]
    return macd


if sma_check:
    sma = sma(stock_data)
    st.subheader("Simple Moving Average")
    st.write(sma)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sma['Date'], y=sma['SMA'], name = "sma_plot"))
    fig.layout.update(title_text = "SMA", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)



if ema_check:
    ema = ema(stock_data)
    st.subheader("Exponential Moving Average")
    st.write(ema)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ema['Date'], y=ema['EMA'], name = "ema_plot"))
    fig.layout.update(title_text = "EMA", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)



if macd_check:
    st.subheader("Moving Average Convergence/Divergence")
    macd = macd(stock_data)
    st.write(macd)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd['Date'], y=macd['MACD'], name = "macd_plot"))
    fig.layout.update(title_text = "MACD", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
