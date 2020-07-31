import pandas as pd
import datetime
from constants import TICK_PRICE_FIELD, TICK_VOLUME_FIELD, \
    OPEN, HIGH, LOW, CLOSE, VOLUME, TRADES, VWAP
import logging

def load_data():
    data_df = pd.read_csv('./data/tick_data.csv', index_col='Date')
    #to local date time
    data_df.index = data_df.index.map(datetime.datetime.fromtimestamp)
    return data_df

def cleanData(data_df, field, checklist=(NAN, NON_POSTIVE, OUTLIER), **kwargs):
    #check nan
    if NAN in checklist:
        nan_df = data_df[data_df[field].isnull()]
        if len(nan_df) > 0:
            logging.log(logging.WARNING, field + " Data contains na, data will be forward filled")
            data_df[field] = data_df[field].ffill()

    #check non positive
    if NON_POSTIVE in checklist:
        non_postive_df = data_df[data_df[field]<=0]
        if len(non_postive_df) > 0:
            logging.log(logging.WARNING, field + " Data contains non positive, nonpositive is going to be dropped")
            data_df = data_df.drop(non_postive_df.index)
    #check outlier
    if OUTLIER in checklist:
        outlier_df = get_outlier_std(data_df[field], **kwargs)
        if  len(outlier_df) > 0:
            logging.log(logging.WARNING, field + " Data contains outlier, outlier is going to be dropped")
            data_df = data_df.drop(outlier_df.index)
    return data_df

def get_outlier_std_rolling(data_series,window=100, n_std = 3):
    ds_last_mean = data_series.rolling(window=window).mean()
    ds_last_std = data_series.rolling(window=window).std()
    return data_series[((data_series - ds_last_mean).abs() > n_std * ds_last_std)]

def get_outlier_std(data_series, n_std = 3):
    ds_last_mean = data_series.mean()
    ds_last_std = data_series.std()
    return data_series[((data_series - ds_last_mean).abs() > n_std * ds_last_std)]

def get_outlier_boxplot(data_series,window=100, n_iqr = 3):
    q1 = data_series.rolling(window=window).quantile(0.25)
    q3 = data_series.rolling(window=window).quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-n_iqr*iqr
    fence_high = q3+n_iqr*iqr
    return data_series[(data_series < fence_low) | (data_series > fence_high)]

def bar_from_tick(data_df, freq = '1min', non_trading_hours = None):
    # get open high low close volume vwap
    resampled_data_df = data_df.groupby(pd.Grouper(freq=freq)).agg({TICK_PRICE_FIELD:['first', 'max', 'min', 'last'],
                                                                    TICK_VOLUME_FIELD:['sum', 'count']
                                                                  }).set_axis([OPEN, HIGH, LOW, CLOSE, VOLUME, TRADES], axis=1)

    resampled_data_df[VWAP] = data_df.groupby(pd.Grouper(freq=freq)).apply(
        lambda data: (data.Price * data.Size).sum() / data.Size.sum() if data.Size.sum() > 0 else 0)
    if non_trading_hours is not None:
        for (start_time, end_time) in non_trading_hours:
            resampled_data_df = resampled_data_df.drop(resampled_data_df.between_time(start_time, end_time, include_end=False).index)
    return resampled_data_df

def bar_from_high_resolution(data_df, freq = '5min'):
    resampled_data_df = data_df.groupby(pd.Grouper(freq=freq)).agg({OPEN:'first',
                                                                    HIGH:'max',
                                                                    CLOSE: 'last',
                                                                    LOW:'min',
                                                                    VOLUME:'sum',
                                                                    TRADES:'sum'
                                                                  })
    resampled_data_df[VWAP] = data_df.groupby(pd.Grouper(freq=freq)).apply(
        lambda data: (data.vwap * data.volume).sum() / data.volume.sum() if data.volume.sum() > 0 else 0 )
    resampled_data_df = resampled_data_df.dropna()
    return resampled_data_df

if __name__ == '__main__':
    ls = list(range(0,500))
    test_data = pd.DataFrame(dict(
        a = ls
    ))
    test = get_outlier_std_rolling(test_data['a'],window=100, n_std=3)
