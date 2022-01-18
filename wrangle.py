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
    nem_usd = yf.download("XEM-USD", period='max')
    
    # pull in all daily data for HOT1-USD
    holo_usd = yf.download('HOT1-USD', period='max')
    
    # pull in hourly data since Dec 1, 2021 for XEM-USD
    nem_usd_hr = yf.download("XEM-USD", start="2021-12-01", end="2022-01-16", interval='1h')
    
    # pull in hourly data since Dec 1, 2021 for XEM-USD
    holo_usd_hr = yf.download("HOT1-USD", start="2021-12-01", end="2022-01-16", interval='1h')
    
    return nem_usd, holo_usd, nem_usd_hr, holo_usd_hr


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