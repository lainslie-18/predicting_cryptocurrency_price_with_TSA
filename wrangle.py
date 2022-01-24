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



def transform_hourly_data(nem_hr_train, nem_hr_validate, holo_hr_train, holo_hr_validate):
    # create date ranges with hourly frequency to merge with NEM and HOLO hourly data
    nem_range = pd.date_range(start='2021-12-23 23:00:00+0000', end='2022-01-07 02:00:00+0000', freq='H')
    holo_range = pd.date_range(start='2021-12-23 23:00:00+0000', end='2022-01-07 02:00:00+0000', freq='H')
    # turn my_range into a dataframe with a date column
    nem_range = pd.DataFrame(nem_range)
    holo_range = pd.DataFrame(holo_range)

    nem_range = nem_range.rename(columns={0:'Date'})
    holo_range = holo_range.rename(columns={0:'Date'})
    # set date column as index
    nem_range = nem_range.set_index('Date')
    holo_range = holo_range.set_index('Date')
    # merge my_range and nem_hr_validate
    nem_hr_validate2 = nem_range.merge(nem_hr_validate, how='left', left_index=True, right_index=True)
    holo_hr_validate2 = holo_range.merge(holo_hr_validate, how='left', left_index=True, right_index=True)
    # use forward fill to fill in missing values
    nem_hr_validate2.ffill(axis=0, inplace=True)
    holo_hr_validate2.ffill(axis=0, inplace=True)

    nem_hr_validate = nem_hr_validate2.copy()
    holo_hr_validate = holo_hr_validate2.copy()
    
    nem_hr_validate.index = nem_hr_validate.index.tz_localize(None)
    holo_hr_validate.index = holo_hr_validate.index.tz_localize(None)
    nem_hr_train.index = nem_hr_train.index.tz_localize(None)
    holo_hr_train.index = holo_hr_train.index.tz_localize(None)
    
    return nem_hr_train, nem_hr_validate, holo_hr_train, holo_hr_validate



# plot original and predicted values
def plot_and_eval(train_set, validate_set, yhat, model, train_set2, validate_set2, yhat2, model2):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(train_set['Close'], label = 'Train', linewidth = 1)
    validate_set['Close'].plot(ax=ax1, label = 'Validate', linewidth = 1)
    yhat['Close'].plot(ax=ax1, label = 'Prediction', linewidth = 1)
    ax1.set(title=('Actual Closing Price vs ' + model), xlabel='Date', ylabel='Closing Price ($)')
    ax1.legend(['Train', 'Validate', 'Prediction'])
#     plt.xticks(rotation=45)
    ax2.plot(train_set2['Close'], label = 'Train', linewidth = 1)
    validate_set2['Close'].plot(ax=ax2, label = 'Validate', linewidth = 1)
    yhat2['Close'].plot(ax=ax2, label = 'Prediction', linewidth = 1)
    ax2.set(title=('Actual Closing Price vs ' + model2), xlabel='Date', ylabel='Closing Price ($)')
    ax1.legend(['Train', 'Validate', 'Prediction'])
#     plt.xticks(rotation=45)
    plt.tight_layout()
    rmse1 = evaluate(validate_set, yhat)
    rmse2 = evaluate(validate_set2, yhat2)
    print(f'{model} RMSE: {rmse1}')
    print(f'{model2} RMSE: {rmse2}')
    plt.show()


#####################



def append_eval_df(eval_df, train_set, validate_set, yhat, model):
    rmse = evaluate(validate_set, yhat)
    d = pd.DataFrame({'model': [model], 'rmse': [rmse]})
    eval_df = eval_df.append(d, ignore_index = True)
    return eval_df


def predict_evaluate_baseline(train_set, validate_set, train_set2, validate_set2, eval_df):
    # predict using mean of train close data
    model = 'Baseline Prediction - Daily'
    close = round(train_set['Close'].mean(), 5)
    yhat = pd.DataFrame({'Close': [close]}, index = validate_set.index)
    print(f'{model}  = {close}')

    model2 = 'Baseline Prediction - Hourly'
    close2 = round(train_set2['Close'].mean(), 5)
    yhat2 = pd.DataFrame({'Close': [close2]}, index = validate_set2.index)
    print(f'{model2}  = {close2}')

    plot_and_eval(train_set, validate_set, yhat, model, train_set2, validate_set2, yhat2, model2)
    eval_df = append_eval_df(eval_df, train_set, validate_set, yhat, model)
    eval_df = append_eval_df(eval_df, train_set2, validate_set2, yhat2, model2)

    return eval_df




def predict_evaluate_mavg(train_set, validate_set, period, train_set2, validate_set2, period2, eval_df):
    # predict using mean of train close data
    model = '1-Yr Moving Avg'
    close = round(train_set.Close.rolling(period).mean().iloc[-1], 5)
    yhat = pd.DataFrame({'Close': [close]}, index = validate_set.index)
    print(f'{model}  = {close}')

    model2 = '16-Hr Moving Avg'
    close2 = round(train_set2.Close.rolling(period).mean().iloc[-1], 5)
    yhat2 = pd.DataFrame({'Close': [close2]}, index = validate_set2.index)
    print(f'{model2}  = {close2}')

    plot_and_eval(train_set, validate_set, yhat, model, train_set2, validate_set2, yhat2, model2)
    eval_df = append_eval_df(eval_df, train_set, validate_set, yhat, model)
    eval_df = append_eval_df(eval_df, train_set2, validate_set2, yhat2, model2)

    return eval_df

# def predict_evaluate_16hr_mavg(train_set, validate_set, eval_df):
#     # predict using mean of train close data
#     model = '16-Hr Moving Avg'
#     close = round(train_set.Close.rolling(16).mean().iloc[-1], 5)
#     yhat = pd.DataFrame({'Close': [close]}, index = validate_set.index)
#     print(f'{model}  = {close}')

#     plot_and_eval(train_set, validate_set, yhat, model)
#     eval_df = append_eval_df(eval_df, train_set, validate_set, yhat, model)

#     return eval_df

def predict_evaluate_holts_linear(train_set, validate_set, sm_level, sm_slope, train_set2, validate_set2, sm_level2, sm_slope2, eval_df):
    # predict using mean of train close data
    model = 'Holt\'s Linear Trend - Daily'
    close = Holt(train_set.Close, exponential = False)
    close = close.fit(smoothing_level = sm_level, smoothing_slope = sm_slope, optimized = True)
    yhat = pd.DataFrame(round(close.predict(start = validate_set.index[0], end = validate_set.index[-1]), 5), columns=['Close'])

    model2 = 'Holt\'s Linear Trend - Hourly'
    close2 = Holt(train_set2.Close, exponential = False)
    close2 = close2.fit(smoothing_level = sm_level2, smoothing_slope = sm_slope2, optimized = True)
    yhat2 = pd.DataFrame(round(close2.predict(start = validate_set2.index[0], end = validate_set2.index[-1]), 5), columns=['Close'])

    plot_and_eval(train_set, validate_set, yhat, model, train_set2, validate_set2, yhat2, model2)
    eval_df = append_eval_df(eval_df, train_set, validate_set, yhat, model)
    eval_df = append_eval_df(eval_df, train_set2, validate_set2, yhat2, model2)

    return eval_df

def predict_evaluate_arima(train_set, validate_set, order, train_set2, validate_set2, order2, eval_df):
    # predict using mean of train close data
    model = 'ARIMA - Daily'
    close = ARIMA(train_set.Close, order=order)
    close = close.fit()
    yhat = pd.DataFrame(round(close.predict(start = validate_set.index[0], end = validate_set.index[-1]), 5), columns=['Close'])

    model2 = 'ARIMA - Hourly'
    close2 = ARIMA(train_set2.Close, order=order)
    close2 = close2.fit()
    yhat2 = pd.DataFrame(round(close2.predict(start = validate_set2.index[0], end = validate_set2.index[-1]), 5), columns=['Close'])

    plot_and_eval(train_set, validate_set, yhat, model, train_set2, validate_set2, yhat2, model2)
    eval_df = append_eval_df(eval_df, train_set, validate_set, yhat, model)
    eval_df = append_eval_df(eval_df, train_set2, validate_set2, yhat2, model)

    return eval_df
