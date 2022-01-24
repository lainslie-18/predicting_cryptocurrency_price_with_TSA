import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf


from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA

# create function to pull in data

def get_crypto_data():
    '''
    This function downloads data from Yahoo Finance using yfinance and returns four dataframes
    '''
    
    # pull in all daily data for XEM-USD
    nem = yf.download("XEM-USD", period='max', progress=False)

    # pull in hourly data since Dec 1, 2021 for XEM-USD
    nem_hr = yf.download("XEM-USD", start="2021-12-01", end="2022-01-16", interval='1h', progress=False)
    
    # pull in all daily data for HOT1-USD
    holo = yf.download('HOT1-USD', period='max',progress=False)
    
    # pull in hourly data since Dec 1, 2021 for XEM-USD
    holo_hr = yf.download("HOT1-USD", start="2021-12-01", end="2022-01-16", interval='1h', progress=False)
    
    return nem, nem_hr, holo, holo_hr


# create a function that splits data
def split_time_series_data(df):
    '''
    This function takes in a dataframe, does a 50/30/20 split, and returns three dataframes for training, validating, and testing
    '''
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    validate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]
    
    return train, validate, test


def split_crypto_data(nem, nem_hr, holo, holo_hr):

    train1, validate1, test1 = split_time_series_data(nem)
    train2, validate2, test2 = split_time_series_data(nem_hr)
    train3, validate3, test3 = split_time_series_data(holo)
    train4, validate4, test4 = split_time_series_data(holo_hr)

    # creates a list of the split dataframes for each ticker symbol
    crypto1 = [train1, validate1, test1, train2, validate2, test2]
    crypto2 = [train3, validate3, test3, train4, validate4, test4]

    # creates a list of the two lists
    cryptocurrencies = [crypto1, crypto2]

    return cryptocurrencies


def prep_crypto_data():

    # pull in data
    nem, nem_hr, holo, holo_hr = get_crypto_data()

    # drop all columns except Close as target variable
    nem = pd.DataFrame(nem.Close)
    nem_hr = pd.DataFrame(nem_hr.Close)
    holo = pd.DataFrame(holo.Close)
    holo_hr = pd.DataFrame(holo_hr.Close)

    # list of nem_train, nem_validate, nem_test, nem_hr_train, nem_hr_validate, nem_hr_test, holo_train, holo_validate, holo_test, holo_hr_train, holo_hr_validate, holo_hr_test,
    cryptocurrencies = split_crypto_data(nem, nem_hr, holo, holo_hr)

    return cryptocurrencies


# compute rmse to evaluate model
def evaluate(validate_set, yhat):
    rmse = round(sqrt(mean_squared_error(validate_set['Close'], yhat['Close'])), 5)
    return rmse


# plot original and predicted values
def plot_and_eval(train_set, validate_set, yhat, model):
    plt.plot(train_set['Close'], label = 'Train', linewidth = 1)
    plt.plot(validate_set['Close'], label = 'Validate', linewidth = 1)
    plt.plot(yhat['Close'], label = 'Prediction', linewidth = 1)
    plt.title('Actual Closing Price vs ' + model)
    plt.legend()
    plt.xticks(rotation=45)
    rmse = evaluate(validate_set, yhat)
    print(f'{model} RMSE: {rmse}')
    plt.show()

#################

# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(holo_train, label='Train')
# holo_validate.plot(ax=ax1, label='Validate')
# holo_test.plot(ax=ax1, label='Test')
# ax1.set(title='HOLO Daily Closing Price', xlabel='Date', ylabel='Closing Price ($)')
# ax1.legend(['Train', 'Validate', 'Test'])
# ax2.plot(holo_hr_train, label='Train')
# holo_hr_validate.plot(ax=ax2, label='Validate')
# holo_hr_test.plot(ax=ax2, label='Test')
# ax2.set(title='HOLO Hourly Closing Price', xlabel='Date', ylabel='Closing Price ($)')
# ax2.legend(['Train', 'Validate', 'Test'])
# plt.tight_layout();

# # plot original and predicted values
# def plot_and_eval(train_set, validate_set, yhat, model):
#     plt.plot(train_set['Close'], label = 'Train', linewidth = 1)
#     plt.plot(validate_set['Close'], label = 'Validate', linewidth = 1)
#     plt.plot(yhat['Close'], label = 'Prediction', linewidth = 1)
#     plt.title('Actual Closing Price vs ' + model)
#     plt.legend()
#     plt.xticks(rotation=45)
#     rmse = evaluate(validate_set, yhat)
#     print(f'{model} RMSE: {rmse}')
#     plt.show()


#####################



def append_eval_df(eval_df, train_set, validate_set, yhat, model):
    rmse = evaluate(validate_set, yhat)
    d = pd.DataFrame({'model': [model], 'rmse': [rmse]})
    eval_df = eval_df.append(d, ignore_index = True)
    return eval_df


def predict_evaluate_baseline(train_set, validate_set, eval_df):
    # predict using mean of train close data
    model = 'Baseline Prediction'
    close = round(train_set['Close'].mean(), 5)
    yhat = pd.DataFrame({'Close': [close]}, index = validate_set.index)
    print(f'{model}  = {close}')

    plot_and_eval(train_set, validate_set, yhat, model)
    eval_df = append_eval_df(eval_df, train_set, validate_set, yhat, model)

    return eval_df




def predict_evaluate_1yr_mavg(train_set, validate_set, eval_df):
    # predict using mean of train close data
    model = '1-Year Moving Avg'
    close = round(train_set.Close.rolling(365).mean().iloc[-1], 5)
    yhat = pd.DataFrame({'Close': [close]}, index = validate_set.index)
    print(f'{model}  = {close}')

    plot_and_eval(train_set, validate_set, yhat, model)
    eval_df = append_eval_df(eval_df, train_set, validate_set, yhat, model)

    return eval_df

def predict_evaluate_holts_linear(train_set, validate_set, eval_df):
    # predict using mean of train close data
    model = 'Holt\'s Linear Trend'
    close = Holt(train_set.Close, exponential = False)
    close = close.fit(smoothing_level = .2, smoothing_slope = .9, optimized = True)
    yhat = pd.DataFrame(round(close.predict(start = validate_set.index[0], end = validate_set.index[-1]), 5), columns=['Close'])

    plot_and_eval(train_set, validate_set, yhat, model)
    eval_df = append_eval_df(eval_df, train_set, validate_set, yhat, model)

    return eval_df

def predict_evaluate_arima(train_set, validate_set, eval_df):
    # predict using mean of train close data
    model = 'ARIMA'
    close = ARIMA(train_set.Close, order=(5,0,0))
    close = close.fit()
    yhat = pd.DataFrame(round(close.predict(start = validate_set.index[0], end = validate_set.index[-1]), 5), columns=['Close'])

    plot_and_eval(train_set, validate_set, yhat, model)
    eval_df = append_eval_df(eval_df, train_set, validate_set, yhat, model)

    return eval_df
