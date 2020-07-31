import talib
from constants import OPEN, HIGH, LOW, CLOSE, VOLUME

# class Indicator:

def make_ti_factor(data_df, factor_configs):
    ml_fields = []
    for factor_config in factor_configs:
        factor_name= factor_config['name'] + '_'.join([str(x) for x in factor_config['paras'].values()])
        if 'MACD' in factor_name:
            data_df[factor_name],data_df[factor_name+'_S'],data_df[factor_name+'_H'] = \
                eval(factor_config['name']+'(data_df, **factor_config["paras"])')
            ml_fields.extend([factor_name, factor_name+'_S', factor_name+'_H'])
        else:
            data_df[factor_name] = eval(factor_config['name']+'(data_df, **factor_config["paras"])')
            ml_fields.append(factor_name)
    return ml_fields, data_df

def RSI(data_df, **kwargs):
    return talib.RSI(data_df[CLOSE], timeperiod=kwargs.get('timeperiod', 14))

def MOM(data_df,**kwargs):
    return talib.MOM(data_df[CLOSE], timeperiod=kwargs.get('timeperiod', 10))

def MACD(data_df, **kwargs):
    return talib.MACD(data_df[CLOSE],fastperiod=kwargs.get('fastperiod', 13),
                      slowperiod=kwargs.get('slowperiod', 21),
                      signalperiod=kwargs.get('signalperiod', 8))

def MFI(data_df, **kwargs):
   return talib.MFI(data_df[HIGH],data_df[LOW], data_df[CLOSE],
                                   data_df[VOLUME], timeperiod = kwargs.get('timeperiod', 14))

