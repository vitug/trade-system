# coding: utf-8

# для совместимости с версией 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import pandas as pd
import numpy as np
import numba
import matplotlib.pyplot as plt
import scipy.stats as stat
import statsmodels.api as sm
from IPython.display import display, HTML
import winsound
import os

import gorchakov
import intraday


def beep():
    try:
        winsound.Beep(2000, 200)
    except:
        print('winsound::cannot to beep!')
        pass

def path2hdf(path):
    hdf_path = path.replace('\\','')
    hdf_path = hdf_path.replace('.','')
    return hdf_path

#@numba.autojit
#@numba.jit
class Asset(object):
    #@numba.jit
    def __init__(self):
        self.minutes = []
        # self.path = 'data\\eurusd.txt'
        self.path = 'data\\sber.txt'
        self.hdf_database_path = 'minutes.hd5'
        self.candles = None
        self.statCalculator = gorchakov.Statistics()
        self.conf_int_level = 0.05
        self.model_forecast = None
        self.forecast_coef = 0
        self.forecast_error = [0, 0]
        self.forecast_coef_conf_int = [0, 0]
        self.model_true_forecast = None
        self.forecast_true_coef = 0
        self.forecast_true_error = [0, 0]
        self.forecast_true_coef_conf_int = [0, 0]
        self.forecast_error_level = 0.875  # уровень для вычисления квантиля ошибки прогноза следующего приращения
        self.maxStatisticsWindow = 40
        self.periodStr = 'D'
        self.candle_color_model = None
        self.candle_color_coef = 0
        self.candle_color_level = 0.875  # уровень для вычисления квантиля ошибки прогноза цвета свечи
        self.candle_color_bPlus = 0
        self.candle_color_bMinus = 0
        self.timeDelta = pd.datetools.timedelta(0)  # приращение времени между соседними свечами

    #@numba.void(numba.string)
    #@numba.jit
    #если файл сохранен в hd5 файле, то берем оттуда, если там нет то загружаем из текста
    def load(self, path = None, append = False):
        if not path == None:
            self.path = path
        print(self.path)

        append_mode = append and not len(self.minutes) == 0
        if append_mode:
            minutes_temp = self.minutes.copy()
        in_hdf_path = path2hdf(self.path)
        mode = 'a'
        loaded_from_hdf = False
        if os.path.isfile(self.hdf_database_path):
            try:
                self.minutes = pd.read_hdf(self.hdf_database_path, in_hdf_path)
                loaded_from_hdf = True
                print('data are loaded from hd5 file.')
            except:
                pass
            #print(self.minutes)
        else:
            mode = 'w'
        if not loaded_from_hdf:
            dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d %H%M%S')
            self.minutes = pd.read_csv(self.path, sep=';', encoding='latin1', parse_dates={'date': ['<DATE>', '<TIME>']},
                                       dayfirst=False, index_col=0, date_parser=dateparse)
            self.minutes.columns = ['o', 'h', 'l', 'c', 'v']
            self.minutes['hl'] = (self.minutes.h + self.minutes.l) / 2  # средняя цена
            self.minutes['d'] = np.log(self.minutes.hl / self.minutes.hl.shift(1))
            #пишем в hdf
            self.minutes.to_hdf(self.hdf_database_path, in_hdf_path, format ='table', mode = mode)
        if append_mode:
            self.minutes = minutes_temp.append(self.minutes)
        # этот порядок колонок затем будут использоваться в торговле, после конвертации в массив
        print (self.minutes.info())
        beep()

    #устанавливает индексы минутных данных после конвертацию в массив
    @staticmethod
    def set_columns_indexes(object):
        #[u'o', u'h', u'l', u'c', u'v', u'hl', u'd', u'i']
        #[self.io, self.ih, self.il, self.ic, self.iv, self.ihl, self.ii] = range(7)
        [object.io, object.ih, object.il, object.ic, object.iv, object.ihl, id, object.ii] = range(8)

    #@numba.void(numba.bool_)
    #@numba.jit
    def report(self, printForecastModel=False):
        candles = self.candles
        display(candles.head(10))
        print (candles.info())
        print ('corr d + 1 with 2 * c / (h + l) = ', candles.d.corr(candles.chl.shift(1)))
        print ('corr d = ', candles.d.corr(candles.d.shift(1)))
        print ('corr h = ', candles.dh.corr(candles.dh.shift(1)))
        print ('corr l = ', candles.dl.corr(candles.dl.shift(1)))
        print ('corr с = ', candles.dc.corr(candles.dc.shift(1)))
        print ('corr co and ho+lo = ', candles.color.corr(candles.holo))

        if printForecastModel:
            print ('all points forecast model', self.model_forecast)
        print ('forecast coef all data', self.forecast_coef)
        print ('forecast error', self.forecast_error)

        if printForecastModel:
            print ('true points forecast model', self.model_true_forecast)
        print ('forecast true coef', self.forecast_true_coef)
        print ('forecast true error', self.forecast_true_error)
        print ('candle color coef = ', self.candle_color_coef)
        print ('candle color bPlus = ', self.candle_color_bPlus)
        print ('candle color bMinus = ', self.candle_color_bMinus)
        print ('time delta = ', self.timeDelta)
    
    def custom_resampler(self, data):
        data.o = data.o.first()
        data.h = data.h.max()
        data.l = data.l.min()
        data.c = data.c.last()
        data.hl = data.hl.mean()
        data.v = data.v.sum()
        
    #@numba.jit
    def resample(self, period='D', base = 0):
        # Обозначения
        # hl = (h + l) / 2
        # hl1 = ((h + l) / 2)_{t-1}
        # d = Ln (((h + l) / 2)_{t} / ((h + l) / 2})_{t-1}) или ln(hl / hl1)
        # dp1 = d_{t+1}
        # chl = Ln(c/hl)

        # индекс минутной свечи, делаем здесь потому что после загрузки ряд мог измениться
        # после ресэмпла добавлять или удалять строки нельзя
        self.minutes['i'] = 1
        self.minutes.i = (self.minutes.i.cumsum() - 1).astype('int')

        self.periodStr = period
        # self.statCalculator = gorchakov.Statistics()
        conversion = {'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'hl': 'mean', 'v': 'sum'}
        self.candles = self.minutes.resample(period, base=base).apply(conversion).dropna()
        #self.candles = self.minutes.resample(period, how=conversion, base=base).dropna()

        candles = self.candles  # ссылка на члена класса

        self.timeDelta = candles.iloc[1].name - candles.iloc[0].name
        candles['beginTime'] = candles.index
        candles['nextTime'] = candles.beginTime.shift(-1)  # начало следующей свечи
        candles.set_value(candles.iloc[-1].name, 'nextTime', candles.iloc[-1].beginTime + self.timeDelta)

        candles['i'] = 1
        candles.i = (candles.i.cumsum() - 1).astype('int')

        candles['mean_hl'] = candles.hl  # среднее от середин свечей исходного интервала

        candles['hl'] = (candles.h + candles.l) / 2
        #candles['hl'] = candles.hl # если брать не середину, а сумму минуток, то получается хуже (Сбербанк)
        #print '!hl = m!'
        # (candles.hl - candles.m).describe()
        # candles.hl.corr(candles.m) #ошибка:корреляция между средней ценой и хай плюс лоу 0.9997467470191338 для дней сбербанк 2015 год
        # --
        candles['hl1'] = candles.hl.shift(1)  # хай плюс лоу минус один
        candles['d'] = np.log(candles.hl / candles.hl1)  # разность логарифмов
        candles['dp1'] = candles.d.shift(-1)  # будущее приращение
        candles['d1'] = candles.d.shift(1)  # прошлое приращение
        # --
        candles['chl'] = np.log(candles.c / candles.hl)  # для прогноза следующей хай плюс лоу
        # --
        candles['h1'] = candles.h.shift(1)  # хай минус один
        candles['h2'] = candles.h.shift(2)  # хай минус два, для построения начала трендов
        candles['dh'] = np.log(candles.h / candles.h1)
        candles['dhp1'] = candles.dh.shift(-1)
        # --
        candles['l1'] = candles.l.shift(1)  # лоу минус один
        candles['dl'] = np.log(candles.l / candles.l1)
        candles['dlp1'] = candles.dl.shift(-1)
        # --
        candles['m1'] = candles.mean_hl.shift(1)  # средняя цена минус один
        candles['dm'] = np.log(candles.mean_hl / candles.m1)  # разность логарифмов
        candles['dmp1'] = candles.dm.shift(-1)  # будущее приращение
        # --
        candles['c1'] = candles.c.shift(1)  # закрытие минус один
        candles['dc'] = np.log(candles.c / candles.c1)  # разность логарифмов
        candles['dcp1'] = candles.dc.shift(-1)  # будущее приращение
        # --
        candles['color'] = np.log(candles.c / candles.o)  # цвет свечи
        # --
        # хвосты свечей
        candles['ho'] = np.log(candles.h / candles.o)  # хай к открытию
        candles['lo'] = np.log(candles.l / candles.o)  # лоу к открытию
        candles['holo'] = candles.ho + candles.lo  # сумма логарифмов хвостов
        # --
        # Оценка дисперсии эс-эн
        # var = (1 - a1 ^ 2) * sigma ^ 2
        #model = pd.ols(y = candles.d, x = candles.d.shift(1), intercept=False)
        model = sm.OLS(candles.d, candles.d.shift(1)).fit()
        s = model.resid.std()
        ro = candles.d.corr(candles.d.shift(1))
        var = (1 - ro * ro) * s * s
        self.std_auto_regression_process = np.sqrt(var)
        self.statCalculator.min_down_value = self.std_auto_regression_process / 1
        print ('AR(1) var divider 1')
        #--
        # считаем статистики разладки
        for i in range(3, self.maxStatisticsWindow):
            self.statCalculator.calc_stat1(candles, i)
        # --
        for i in range(4, self.maxStatisticsWindow + 1):
            self.statCalculator.calc_stat2_1(candles, i)
        # --
        # считаем линейный прогноз медианной цены
        # hl+1 = hl exp(a ln(c/hl) + b), где b - ошибка, hl+1 - следующая средняя цена, hl - текущая средняя цена
        #self.model_forecast = pd.ols(y=candles.dp1, x=candles.chl, intercept=False)
        self.model_forecast = sm.OLS(candles.dp1, candles.chl, missing='drop').fit()
        #print(self.model_forecast.params[0])
        self.forecast_coef = self.model_forecast.params[0]
        self.forecast_coef_conf_int = self.model_forecast.conf_int(self.conf_int_level)[0]
        residuals = self.model_forecast.resid
        self.forecast_error = np.array([residuals.quantile(1 - self.forecast_error_level),
                                        residuals.quantile(self.forecast_error_level)])
        # Берем только те точки по которым прогноз исполнился
        true_forecast = candles[np.sign(candles.chl) == np.sign(candles.dp1)]
        #self.model_true_forecast = pd.ols(y=true_forecast.dp1, x=true_forecast.chl, intercept=False)
        self.model_true_forecast = sm.OLS(true_forecast.dp1, true_forecast.chl, missing='drop').fit()
        self.forecast_true_coef_conf_int = self.model_true_forecast.conf_int(self.conf_int_level)[0]
        self.forecast_true_coef = self.model_true_forecast.params[0]
        residuals = self.model_true_forecast.resid
        # по квантилям определяем границы прогноза
        self.forecast_true_error = np.array([residuals.quantile(1 - self.forecast_error_level),
                                             residuals.quantile(self.forecast_error_level)])
        # считаем нижнюю и вернюю границы прогноза с учетом ошибки
        candles['fore_high'] = self.forecast_coef * candles.chl + self.forecast_error[1]
        candles['fore_low'] = self.forecast_coef * candles.chl + self.forecast_error[0]
        # Зависимость цвета свечи от суммы хвостов
        # свеча будет белого цвета если
        # h > o^2 / l exp(- b+ / a) где b+ минимальное значение ошибки
        # свеча будет черного цвета если
        # l < o^2 / h exp(- b- / a) где b- максимальное значение ошибки
        #self.candle_color_model = pd.ols(y=candles.color, x=candles.holo, intercept=False)
        self.candle_color_model = sm.OLS(candles.color, candles.holo, missing='drop').fit()
        self.candle_color_coef = self.candle_color_model.params[0]
        residuals = self.candle_color_model.resid
        self.candle_color_bPlus = residuals.quantile(1 - self.candle_color_level)
        self.candle_color_bMinus = residuals.quantile(self.candle_color_level)
        return self.candles

    def ideal(self):
        candles = self.candles
        candles['long_position'] = 0.
        candles['short_position'] = 0.
        tradePosition = 0
        for index, row in candles.iterrows():
            #long
            if row.d > 0 and row.color > 0:
                tradePosition = 1

            if row.color < 0:
                tradePosition = 0

            if tradePosition > 0:
                candles.set_value(index,'long_position',row.dp1)

        for index, row in candles.iterrows():
            #short
            if row.d < 0 and row.color < 0:
                tradePosition = -1

            if row.color > 0:
                tradePosition = 0

            if tradePosition < 0:
                candles.set_value(index,'short_position',row.dp1)

        candles.long_position.cumsum().plot()
        (-candles.short_position).cumsum().plot()
        ax = plt.gca()
        ax2 = plt.twinx(ax)
        #ax2.plot(candles.hl,'--')
        #candles.hl.plot(secondary_y=True)

    def generate_linear(self, start_data, start_price, delta_array):
        minutes_in_day = 8 * 60
        #start_price = 100.
        #delta = 0.01
        index_days = pd.date_range(start_data, periods = len(delta_array), freq='B')
        #print index_days
        frames = [self.minutes]
        i = 0
        for day in index_days:
            #print day
            delta = delta_array[i]
            end_price = start_price * (1 + delta)
            mean = (end_price - start_price) / (minutes_in_day)
            #print end_price, mean
            minutes = start_price + np.arange(minutes_in_day) * mean
            index = pd.date_range(day, periods=minutes.shape[0], freq='T')
            df = pd.DataFrame({'o':minutes,'h':minutes,'l':minutes,'c':minutes, 'v':1}, index = index,
                              columns=['o','h','l','c','v'])
            df['hl'] = df.h
            df['d'] = np.log(df.h / df.h.shift(1))
            frames.append(df)
            #df.h.plot()
            start_price = end_price
            i += 1
        self.minutes = pd.concat(frames)
        return
        #self.minutes.hl.plot()

    def generate(self):
        mean = 0.01
        up = np.zeros(4)
        up = up + mean
        down = np.zeros(4)
        down = down - mean
        deltas = np.concatenate([up, down, up, down])
        self.generate_linear('1/1/2000', 100, deltas)


