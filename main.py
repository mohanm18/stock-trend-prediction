# !pip install yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import pandas_datareader as data

from pandas_datareader import data as pdr
import streamlit as st
import yfinance as yf
yf.pdr_override()
# y_symbols = ['SCHAND.NS', 'TATAPOWER.NS', 'ITC.NS']
from datetime import datetime
from keras.models import load_model 



startdate = datetime(2010,1,1)
enddate = datetime(2019,12,31)

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker [ex: TATAPOWER.NS, AMZN, AMD, TSLA......]', 'TATAPOWER.NS')
df = pdr.get_data_yahoo(user_input, start=startdate, end=enddate)

# Describing the data

st.subheader('Data from 2010-2019')
st.write(df.describe())

# Visualisations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100ma')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, label = 'With 100ma')
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)



st.subheader('Closing Price vs Time Chart with 100ma & 200ma')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, label = 'With 100ma')
plt.plot(ma200, label = 'With 200ma')
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)

# Again splitting data into training and testing and scaling it
# To print graph of just last few days and not the all days

# splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

# Values ranged between 0-1, to feed to model and train efficiently

data_training_array = scaler.fit_transform(data_training)


# Model already trained so no need of x_train and y_train

# # Splitting data into x_train and y_train
# x_train = []
# # 100ma 1st 100days are xtrain 
# # same as for 200ma
# y_train = []
# # 101th day will be y_train\


# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100: i])
#     y_train.append(data_training_array[i, 0])
    
# x_train, y_train = np.array(x_train), np.array(y_train)

# load the model

model = load_model('keras_model.h5')

# testing part
past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predictions

y_predicted = model.predict(x_test)


# All above values are scaled down
# And we need to scale up those values
scaler = scaler.scale_
#above gives factor by which above values were scaled down 

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# Final Graph

st.subheader('Predicted vs Original')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



