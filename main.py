import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

custom_stock = st.text_input('Enter stock symbol (e.g., TSLA)')
stock_symbol = custom_stock.upper() if custom_stock else None

if stock_symbol:
    try:
        stock_info = yf.Ticker(stock_symbol).info

        st.subheader(f"{stock_info['shortName']} ({stock_symbol})")
        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")

        # Fetch current price
        current_price_data = yf.Ticker(stock_symbol).history(period='1d')
        if not current_price_data.empty:
            current_price = current_price_data.iloc[-1]['Close']
            st.write(f"**Current Price:** {stock_info.get('currency', '')} {current_price}")
        else:
            st.warning("Current price information is not available for this stock.")

        st.write(f"**Market Cap:** {stock_info.get('currency', '')} {stock_info.get('marketCap', 'N/A')}")
        st.write(f"**Description:** {stock_info.get('longBusinessSummary', 'N/A')[:400]}...")  # Limiting to 400 characters

        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365

        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data

        data_load_state = st.text('Loading data...')
        data = load_data(stock_symbol)
        data_load_state.text('Loading data... done!')

        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)

        # Calculate profit/loss and recommendation
        current_price = data.iloc[-1]['Close']
        initial_price = data.iloc[0]['Close']
        price_difference = current_price - initial_price

        if price_difference > 0:
            st.success(f"The stock has shown a profit of {price_difference:.2f} USD. You may consider buying.")
        elif price_difference < 0:
            st.error(f"The stock has shown a loss of {abs(price_difference):.2f} USD. You may consider not buying.")
        else:
            st.warning("The stock price remains unchanged. Consider evaluating other factors before making a decision.")
    except Exception as e:
        st.error(f"An error occurred while fetching stock information: {str(e)}")
else:
    st.warning("Please enter a valid stock symbol.")
