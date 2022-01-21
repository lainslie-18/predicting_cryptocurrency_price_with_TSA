import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

# create function to pull in data

def get_crypto_data():
    '''
    This function downloads data from Yahoo Finance using yfinance and returns four dataframes
    '''
    
    # pull in all daily data for XEM-USD
    nem = yf.download("XEM-USD", period='max')

    # pull in hourly data since Dec 1, 2021 for XEM-USD
    nem_hr = yf.download("XEM-USD", start="2021-12-01", end="2022-01-16", interval='1h')
    
    # pull in all daily data for HOT1-USD
    holo = yf.download('HOT1-USD', period='max')
    
    # pull in hourly data since Dec 1, 2021 for XEM-USD
    holo_hr = yf.download("HOT1-USD", start="2021-12-01", end="2022-01-16", interval='1h')
    
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
    cryptocurrencies = split_crypto_data(nem, holo, nem_hr, holo_hr)

    return cryptocurrencies