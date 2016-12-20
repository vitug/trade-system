# coding: utf-8
# для совместимости с версией 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import intraday
import gorchakov
import stocks

class Trader(object):
    def __init__(self):
        """

        :rtype: day trader object
        """
        self.Position = 0
        self.nextLongPrice = np.nan
        self.nextShortPrice = np.nan
        self.longPrice = np.nan
        self.shortPrice = np.nan
        self.row = None
        self.col_long = 'long_position'
        self.col_short = 'short_position'
        # self.index = None
        # self.candles = None
        self.commission = 0.002

    def Trade(self, newPosition, candles, row, index):
        # если позиция была открыта на прошлом шаге, то это цена открытия позиции или медиана если на прошлом шаге позицию держали
        candles.set_value(index, self.col_long + '_open', self.nextLongPrice)
        candles.set_value(index, self.col_short + '_open', self.nextShortPrice)
        self.row = row

        if not self.Position == newPosition:
            self.Close()
            if newPosition == 1:
                self.Buy(np.max([row.hl, row.o]))
            if newPosition == -1:
                self.Sell(np.min([row.hl, row.o]))
        else:
            self.Hold()

        self.Position = newPosition
        # если позиция была закрыта на этом шаге, то это цена закрытия, если позиция сохраняется, то это медиана свечи
        candles.set_value(index, self.col_long + '_close', self.longPrice)
        candles.set_value(index, self.col_short + '_close', self.shortPrice)

    def Buy(self, Price):
        # print " ###Buy"
        self.nextLongPrice = Price

    def Sell(self, Price):
        # print " ###Sell"
        self.nextShortPrice = Price

    def Hold(self):
        # print " ###Hold"
        self.longPrice = np.nan
        self.nextLongPrice = np.nan
        self.shortPrice = np.nan
        self.nextShortPrice = np.nan
        if self.Position == 1:
            self.longPrice = self.nextLongPrice = self.row.hl
        if self.Position == -1:
            self.shortPrice = self.nextShortPrice = self.row.hl

    def Close(self):
        # print " ###Close"
        if self.Position == 1:
            self.longPrice = np.min([self.row.hl, self.row.o])
            self.nextLongPrice = np.nan
            self.shortPrice = np.nan
        if self.Position == -1:
            self.shortPrice = np.max([self.row.hl, self.row.o])
            self.nextShortPrice = np.nan
            self.longPrice = np.nan

class TraderBase(object):
    def __init__(self):
        self.position = 0
        pass

    def start(self, row):  # начало торгового такта
        pass

    def end(self):  # завершение торгового такта
        pass

    def reverse_at_price(self, price):
        self.position = - self.position
        return True

    def market_price(self):
        return 1

    def reverse_at_market(self):
        self.reverse_at_price(self.market_price())

    def open_at_price(self, position, price):
        self.position = position
        return True

    def open_at_market(self, position):
        self.open_at_price(position, self.market_price())

    def hold(self):  # пока не нужно
        pass

    def close_at_market(self):
        self.close_at_price(self.market_price())

    def close_at_price(self, price):
        self.position = 0
        return True

    def calc_position(self, position):
        pass

    def calc(self):
        self.calc_position(1)
        self.calc_position(-1)

    def equity_plot(self, log = True, label_suffix = ''):
        pass

#pandas
class Trader2:
    def __init__(self, candles):
        self.position = 0
        self.row = None
        self.position_open_index = None
        self.col_long = 'l_'
        self.col_short = 'sh_'
        # self.candles = pd.DataFrame(candles[['o','h','l','c','hl']],index = candles.index)
        self.candles = candles.copy()
        defaul_value = 0.
        self.candles['l_open'] = defaul_value
        self.candles['l_hold'] = defaul_value
        self.candles['l_close'] = defaul_value
        self.candles['sh_open'] = defaul_value
        self.candles['sh_hold'] = defaul_value
        self.candles['sh_close'] = defaul_value
        self.candles['l_equity'] = defaul_value
        self.candles['sh_equity'] = defaul_value
        self.candles['l_log_equity'] = defaul_value
        self.candles['sh_log_equity'] = defaul_value
        self.index = None
        # self.candles = None

    def start(self, row):  # начало торгового такта
        self.row = row
        self.index = row.name
        # self.candles.set_value(index, self.col_long + '_close', self.longPrice)
        # self.candles.set_value(index, self.col_short + '_close', self.shortPrice)

    def end(self):  # завершение торгового такта
        pass
        if self.position != 0 and self.position_open_index != self.index:  # позиция отрыта не на текущем шаге
            self.hold()

    def reverse_at_price(self, price):
        if self.position == 0:
            raise Exception('position not open!')
        new_position = - self.position
        opened = False
        if self.position != 0:
            closed = self.close_at_price(price)
            if closed:
                opened = self.open_at_price(new_position, price)
        return opened

    def market_price(self):
        return self.row.hl

    def reverse_at_market(self):
        self.reverse_at_price(self.market_price())

    def column(self, position, suffix):
        if position > 0:
            return 'l_' + suffix
        if position < 0:
            return 'sh_' + suffix
        if position == 0:
            raise Exception('position = 0!')

    def open_at_price(self, position, price):
        # print " ###open at price", position, price
        if self.row.h < price or self.row.l > price:
            return False
        if self.position != 0:
            raise Exception('Position already open!')
        self.position = position
        self.position_open_index = self.index
        self.candles.set_value(self.index, self.column(self.position, 'open'), price)
        return True

    def open_at_market(self, position):
        # print " ###open at market", position
        self.open_at_price(position, self.market_price())

    def hold(self):  # пока не нужно
        # print " ###Hold"
        pass
        if not self.position == 0:
            self.candles.set_value(self.index, self.column(self.position, 'hold'), self.row.hl)

    def close_at_market(self):
        # print " ###close at market", position
        self.close_at_price(self.market_price())

    def close_at_price(self, price):
        # print " ###close at price", position, price
        if not (self.row.h >= price and self.row.l <= price):
            return False
        if self.position == 0:
            raise Exception('Position already closed!')
        self.candles.set_value(self.index, self.column(self.position, 'close'), price)
        if self.position_open_index == 0:
            raise Exception('open index = 0!')
        # устанавливаем цену hold
        self.candles.ix[self.position_open_index:self.index,
                        self.column(self.position, 'hold')] = self.candles.hl  # заполняем колонку hold
        self.position = 0
        self.position_open_index = None
        return True

    def calc_position(self, position):
        if not np.abs(position) == 1:
            raise Exception('position abs != 1')
        df = self.candles
        diff = df[df[self.column(position, 'open')] > 0].shape[0] - df[df[self.column(position, 'close')] > 0].shape[0]
        if diff:
            print ('last position is not closed!', position, diff)
            i = int(df[df[self.column(position, 'open')] > 0].ix[-1].i)
            # закрываем позицию по цене хл следующей свечи
            if i < df.shape[0] - 1:
                print ('closing on next candle...')
                df.ix[i + 1, self.column(position, 'close')] = df.ix[i + 1].hl
        df.ix[df[self.column(position, 'open')] > 0, self.column(position, 'hold')] = 0
        df.ix[df[self.column(position, 'close')] > 0, self.column(position, 'hold')] = 0
        df[self.column(position, 'close_prev')] = df[self.column(position, 'close')].shift(-1)
        df[self.column(position, 'hold_prev')] = df[self.column(position, 'hold')].shift(-1)
        # абсолютная доходность
        df[self.column(position, 'd')] = (df[self.column(position, 'close_prev')] - \
                                          (df[self.column(position, 'hold')] - df[self.column(position, 'hold_prev')]) - \
                                          df[self.column(position, 'open')]).shift(1)
        df[self.column(position, 'log_d')] = 0.
        # логарифмическая доходность
        df[self.column(position, 'log_d')] = np.log(
            (df[self.column(position, 'close_prev')] + df[self.column(position, 'hold_prev')]) / \
            (df[self.column(position, 'hold')] + df[self.column(position, 'open')])).fillna(0).shift(1)
        df[self.column(position, 'log_d')].replace([-np.inf, np.inf], 0, inplace=True)
        df[self.column(position, 'equity')] = position * df[self.column(position, 'd')].cumsum()
        df.ix[0, self.column(position, 'equity')] = 0
        df[self.column(position, 'log_equity')] = position * df[self.column(position, 'log_d')].cumsum()
        df.ix[0, self.column(position, 'log_equity')] = 0

    def calc(self):
        self.calc_position(1)
        self.calc_position(-1)

    def equity_plot(self, log = True, label_suffix = ''):
        if log:
            eq_name = 'log_equity'
        else:
            eq_name = 'equity'
        plt.plot(self.candles[self.column(1, eq_name)], label = 'long {}'.format(label_suffix))
        #plt.plot(self.candles[self.column(-1, eq_name)], label = 'short {}'.format(label_suffix))
        plt.show()
        plt.draw()

#numpy
class Trader3:
    def __init__(self, candles):
        stocks.Asset.set_columns_indexes(self)
        self.index_array = candles.index.values
        self.position = 0
        self.row = None
        self.position_open_index = None
        self.col_long = 'l_'
        self.col_short = 'sh_'
        # self.candles = pd.DataFrame(candles[['o','h','l','c','hl']],index = candles.index)
        self.candles = candles.copy()
        defaul_value = 0.
        self.candles['l_open'] = defaul_value
        self.candles['l_hold'] = defaul_value
        self.candles['l_close'] = defaul_value
        self.candles['sh_open'] = defaul_value
        self.candles['sh_hold'] = defaul_value
        self.candles['sh_close'] = defaul_value
        self.candles['l_equity'] = defaul_value
        self.candles['sh_equity'] = defaul_value
        self.candles['l_log_equity'] = defaul_value
        self.candles['sh_log_equity'] = defaul_value
        self.index = None
        self.commission = 0.002 # процент комиссии на сделку
        self.calc_commission = True
        # self.candles = None

    def start(self, row):  # начало торгового такта
        self.row = row
        self.index = self.index_array[row[self.ii]]
        # self.candles.set_value(index, self.col_long + '_close', self.longPrice)
        # self.candles.set_value(index, self.col_short + '_close', self.shortPrice)

    def end(self):  # завершение торгового такта
        pass
        if self.position != 0 and self.position_open_index != self.index:  # позиция отрыта не на текущем шаге
            self.hold()

    def reverse_at_price(self, price):
        if self.position == 0:
            raise Exception('position not open!')
        new_position = - self.position
        opened = False
        if self.position != 0:
            closed, stop_price = self.stop_at_price(price)
            if closed:
                opened, open_price = self.open_at_price(new_position, stop_price)
        return opened

    def market_price(self):
        return self.row[self.ihl]

    def reverse_at_market(self):
        self.reverse_at_price(self.market_price())

    def column(self, position, suffix):
        if position > 0:
            return 'l_' + suffix
        if position < 0:
            return 'sh_' + suffix
        if position == 0:
            raise Exception('position = 0!')

    def open_at_price(self, position, price):
        # print " ###open at price", position, price
        if self.position != 0:
            raise Exception('Position already open!')
        real_price = 0
        if self.row[self.ih] >= price and self.row[self.il] <= price:
            #цена попала в интервал свечи
            real_price = price
        else:
            if position > 0:
                if self.row[self.il] > price:
                    #цена уже выше цены по которой мы хотели купить
                    real_price = self.row[self.il]
                else:
                    return False, 0
            if position < 0:
                if self.row[self.ih] < price:
                    #цена уже ниже цены по которой мы хотели продать
                    real_price = self.row[self.ih]
                else:
                    return False, 0
        if np.abs(np.log(real_price/price)) > 0.005:
            raise Exception('big difference!')
        if real_price == 0:
            raise Exception('price is zero!')
        self.position = position
        price_with_commission = real_price
        if self.calc_commission:
            price_with_commission = real_price + position * self.commission * real_price
        self.position_open_index = self.index
        self.candles.set_value(self.index, self.column(self.position, 'open'), price_with_commission)
        return True, real_price

    def open_at_market(self, position):
        # print " ###open at market", position
        self.open_at_price(position, self.market_price())

    def hold(self):  # пока не нужно
        # print " ###Hold"
        pass
        if not self.position == 0:
            self.candles.set_value(self.index, self.column(self.position, 'hold'), self.row[self.ihl])

    def close_at_market(self):
        # print " ###close at market", position
        self.stop_at_price(self.market_price())

    #def close_at_price(self, price):
    def stop_at_price(self, price): # цена сделки может отличаться от цены стопа из-за разрывов в ценовом ряде
        # print " ###close at price", position, price
        # если позиция лонг и цена ниже цены прайс закрываем по рынку
        # если позиция шорт и цена выше цены прайс закрываем по рынку
        if self.position == 0:
            raise Exception('Position closed!')
        real_price = 0
        # проверяем на попадание в ценовой диапазон, если попали в него, то цена сделки равна цене стопа
        if (self.row[self.ih] >= price and self.row[self.il] <= price):
            real_price = price
        else:
            if self.position > 0:
                #если стоим в лонге, то нет смысла закрывать позицию пока цена выше цены стопа
                if self.row[self.il] > price:
                    return False, 0
                else:
                    # стоп выше чем диапазон свечи, значит был разрыв между свечами
                    # исполняем сделку по цене хая
                    real_price = self.row[self.ih]
            if self.position < 0:
                #если стоим в шорте, то нет смысла закрывать позицию пока цена ниже цены стопа
                if self.row[self.ih] < price:
                    return False, 0
                else:
                    # стоп ниже чем диапазон свечи, значит был разрыв между свечами
                    # исполняем сделку по цене лоя
                    real_price = self.row[self.il]
        if real_price == 0:
            raise Exception('price is zero!')
        price_with_commission = real_price
        if self.calc_commission:
            price_with_commission = real_price - self.position * self.commission * real_price
        self.candles.set_value(self.index, self.column(self.position, 'close'), price_with_commission)
        if self.position_open_index == 0:
            raise Exception('open index = 0!')
        # устанавливаем цену hold
        self.candles.ix[self.position_open_index:self.index,
                        self.column(self.position, 'hold')] = self.candles.hl  # заполняем колонку hold
        self.position = 0
        self.position_open_index = None
        return True, real_price

    def calc_position(self, position):
        if not np.abs(position) == 1:
            raise Exception('position abs != 1')
        df = self.candles
        diff = df[df[self.column(position, 'open')] > 0].shape[0] - df[df[self.column(position, 'close')] > 0].shape[0]
        if diff:
            print ('last position is not closed!', position, diff)
            iOpen = int(df[df[self.column(position, 'open')] > 0].ix[-1].i);
            #iClose = iOpen + 1; msg = 'closing on next candle...'# закрываем позицию по цене хл следующей свечи
            iClose = int(df.shape[0] - 1); msg = 'closing on last candle' # закрываем на последней свече
            df.ix[iOpen:iClose, self.column(position, 'hold')] = df.hl
            if iClose <= df.shape[0] - 1:
                print (msg)
                df.ix[iClose, self.column(position, 'close')] = df.ix[iClose].hl
        df.ix[df[self.column(position, 'open')] > 0, self.column(position, 'hold')] = 0
        df.ix[df[self.column(position, 'close')] > 0, self.column(position, 'hold')] = 0
        df[self.column(position, 'close_prev')] = df[self.column(position, 'close')].shift(-1)
        df[self.column(position, 'hold_prev')] = df[self.column(position, 'hold')].shift(-1)
        # абсолютная доходность
        df[self.column(position, 'd')] = (df[self.column(position, 'close_prev')] - \
                                          (df[self.column(position, 'hold')] - df[self.column(position, 'hold_prev')]) - \
                                          df[self.column(position, 'open')]).shift(1)
        df[self.column(position, 'log_d')] = 0.
        # логарифмическая доходность
        df[self.column(position, 'log_d')] = np.log(
            (df[self.column(position, 'close_prev')] + df[self.column(position, 'hold_prev')]) / \
            (df[self.column(position, 'hold')] + df[self.column(position, 'open')])).fillna(0).shift(1)
        df[self.column(position, 'log_d')].replace([-np.inf, np.inf], 0, inplace=True)
        df[self.column(position, 'equity')] = position * df[self.column(position, 'd')].cumsum()
        df.ix[0, self.column(position, 'equity')] = 0
        df[self.column(position, 'log_equity')] = position * df[self.column(position, 'log_d')].cumsum()
        df.ix[0, self.column(position, 'log_equity')] = 0

    def calc(self):
        self.calc_position(1)
        self.calc_position(-1)

    def equity_plot(self, log = True, label_suffix = ''):
        if log:
            eq_name = 'log_equity'
        else:
            eq_name = 'equity'
        plt.plot(self.candles[self.column(1, eq_name)], label = 'long {}'.format(label_suffix))
        #plt.plot(self.candles[self.column(-1, eq_name)], label = 'short {}'.format(label_suffix))
        #plt.show()
        #plt.draw()

def _calc_equity(asset):
    # пример расчета доходности системы
    df = asset.minutes.copy()
    df['buy'] = 0.
    df['hold'] = 0.
    df['sell'] = 0.
    # df.set_value(df.iloc[0].name, 'buy', 10)
    df.set_value(df.iloc[1].name, 'buy', df.iloc[1].m)
    df.set_value(df.iloc[2].name, 'hold', df.iloc[2].m)
    df.set_value(df.iloc[3].name, 'hold', 10)
    df.set_value(df.iloc[4].name, 'sell', 14)
    # df.set_value(df.iloc[6].name, 'buy', 15)
    df.set_value(df.iloc[6].name, 'hold', 16)
    df.set_value(df.iloc[7].name, 'hold', 17)
    df['sm1'] = df.sell.shift(-1)
    df['hm1'] = df.hold.shift(-1)
    # абсолютная доходность
    df['d'] = (df.sm1 - (df.hold - df.hm1) - df.buy).shift(1)
    df['sh'] = 0.
    # логарифмическая доходность
    df.sh = np.log((df.sm1 + df.hm1) / (df.hold + df.buy)).fillna(0).shift(1)
    df.sh = df.sh.replace([-np.inf, np.inf], 0)
    # df.sh = df.sh.fillna(0)
    # df['d_log'] = df.sell.shift(-1) / df.hold * df.hold.shift(-1) / df.buy).shift(1)
    df['eq'] = df.d.cumsum()
    df['log_eq'] = df.sh.cumsum()
    # пример замены значений в колонке по условию
    # df.ix[df.buy == 0, 'buy'] = 1
    # df.buy.ix[df.buy == 0] = 1 # то же самое
    # df.buy.loc[df.buy == 0] = 1 # то же самое
    # df.ix[df.buy == 0].buy = 1 #так будет изменено значение в копии колонки, а не в самой колонке
    df.head(10)


def _trade_example(asset):
    asset.candles.head()
    df = asset.minutes.ix[0:10]
    tr = Trader2(df)
    i = 0
    for index, row in df.iterrows():
        tr.start(row)
        if i == 0:
            tr.open_at_market(-1)
        if i == 2:
            tr.close_at_market()
        if i == 3:
            tr.open_at_price(1, row.o)
        if i == 5:
            tr.close_at_price(row.c)
        if i == 6:
            tr.open_at_market(-1)
        i += 1
    tr.calc()
    r = tr.candles
    # r.l_equity.ix[0] = 0
    r.l_equity.plot()
    r.sh_equity.plot()
    plt.show()
    r.l_log_equity.plot()
    r.sh_log_equity.plot()
    r.l_equity.head(10)
    r[['l_open', 'l_hold', 'l_close', 'l_d', 'l_log_d', 'sh_open', 'sh_hold', 'sh_close', 'sh_d', 'sh_log_d', ]].head(
        10)
    # asset = Asset()
    # asset.load()
    # asset.resample('D')
    # asset.report()
    # tr = Trader()
    # tr2 = Trader2(asset.minutes)

class IntradayQ():
    def __init__(self, t_level, asset):
        self.init(None, None, True)
        self.asset = asset
        self.t_disorder_level = t_level
        self.ds = gorchakov.DisorderSeacher(asset.statCalculator)
        self.trader = Trader2(asset.minutes)
        self.disorder = gorchakov.Disorder()
        self.disorder.t_level1 = .75;
        self.disorder.t_level2 = .75;
        self.extra_row = None #текущая свеча большего периода
        self.row = None #минутная свеча
        self.position = 0
        self.forecast_coef = asset.forecast_coef
        self.forecast_error = asset.forecast_error
        self.windowSize = 3

    #запускается вначале периода
    def init(self, extra_row, candles, firstRun = False):
        self.candles = candles
        self.h = 0.
        self.l = 0.
        self.o = 0.
        self.hl = 0.
        self.d = 0.
        self.i = 0  #индекс свечи
        self.color_forecast = 0.
        self.extra_row = extra_row
        self.price_buy = 0
        self.price_sell = 0
        #self.windowSize = windowSize
        if not firstRun:
            self.prev_hl = extra_row.hl1
            self.o = extra_row.o
            self.d_level_plus = extra_row['dcr1_plus{}_{}'.format(self.windowSize, self.t_disorder_level)] #приращение разладка вверх первая статистика
            self.d_level_minus = extra_row['dcr1_minus{}_{}'.format(self.windowSize, self.t_disorder_level)] #приращение разладка вниз первая статистика
            self.hl_plus = self.prev_hl * np.exp(self.d_level_plus) #цена разладки вверх M в терминах АГ
            self.hl_minus = self.prev_hl * np.exp(self.d_level_minus) #цена разладки вниз М в терминах АГ
            #print ' d levels', self.d_level_plus, self.d_level_minus
            #print ' hl levels', self.hl_plus, self.hl_minus

    def check_candle(self, intra_row, force = False):
        if self.i == 0:
            force = True  #первая минута, устанавливаем размах принудительно
        self.row = intra_row
        self.trader.start(intra_row)#начало торговли на текущем такте
        if force:
            self.h = intra_row.h
            self.l = intra_row.l
            self.new_extremum()

        if intra_row.h > self.h:
            self.h = intra_row.h
            self.new_extremum()

        if intra_row.l < self.l:
            self.l = intra_row.l
            self.new_extremum()
        #self.trader.end() #вроде не нужно пока

        #if self.i == self.candles.shape[0] - 1:
        if self.candles.ix[-1].name == self.candles.ix[self.i].name:
            self.on_period_close()
        self.i += 1

    def on_period_close(self):
        #intraday.trader.close_at_market()
        print (' last intraday candle')
        #расчет цены закрытия при которой будет разладка на следующем шаге
        # f = chl * a + b, здесь f - критический прогноз на разладку chl = ln(c/hl), b - ошибка прогноза
        # c = hl * exp((f - b) / a)
        stat1, disorder, levels = self.asset.statCalculator.calc_stat1(self.extra_row, self.windowSize + 1, 0.01, self.t_disorder_level)
        #self.price_close_forecast_up
        price_close_forecast_up = self.hl * np.exp( (levels[0] - self.forecast_error[0]) / self.forecast_coef )
        price_close_forecast_down = self.hl * np.exp( (levels[1] - self.forecast_error[1]) / self.forecast_coef )
        #price_close_forecast_down = self.hl * np.exp( (levels[1]) / self.forecast_coef )
        #print '  f', price_close_forecast_up, price_close_forecast_down
        #print '  d', self.price_level_up, self.price_level_down
        #print '  c', self.price_color_white, self.price_color_black
        close_buy_level = np.max([self.o, np.min([self.price_level_up, price_close_forecast_up])])
        close_sell_level = np.min([self.o, np.max([self.price_level_down, price_close_forecast_down])])
        print ('  up {:.2f} down {:.2f} open {} close {}'.format(close_buy_level, close_sell_level, self.o, self.row.c))
        if self.trader.position == 0:
            if self.row.c >= close_buy_level:
                print (' @long at close')
                self.trader.open_at_price(1, self.row.c)
            if self.row.c <= close_sell_level:
                print (' @short at close')
                self.trader.open_at_price(-1, self.row.c)

        #разладка
        self.disorder.Check(self.extra_row, self.windowSize)

        if self.disorder.step == 1 and self.disorder.disorder1:
            #print index
            print (' first disorder ', self.windowSize, self.disorder.sign1, self.extra_row.color, self.extra_row.d)

        if self.disorder.step == 2 and self.disorder.disorder2:
            print (' ++second disorder', self.windowSize)
            self.windowSize = 2

        if self.disorder.step == 2 and not self.disorder.disorder2:
            print (' --first disorder not approved')
            pass
        #сбрасываем разладки в исходное состояние на шаге 2
        if self.disorder.step == 2:
            self.disorder.clear()

        self.windowSize += 1

    def new_extremum(self):
        self.hl = (self.h + self.l) / 2
        self.d = np.log(self.hl / self.prev_hl) # текущее приращение логарифмов цен
        # P_up = 2 * M - L
        # P_down = 2 * M - H
        self.price_level_up = 2 * self.hl_plus - self.l #расчетная цена разладки вверх при условии неизменности текущего лоу
        self.price_level_down = 2 * self.hl_minus - self.h #расчетная цена разладки вверх при условии неизменности текущего хая
        #расчет прогнозов белой и черной свечи
        # O+(p) = exp( -h(p) / a) * O^2 / L
        # O-(p) = exp( -l(p) / a) * O^2 / H
        self.price_color_white = np.exp(- self.candle_color_bPlus / self.candle_color_coef) * self.o * self.o / self.l
        self.price_color_black = np.exp(- self.candle_color_bMinus / self.candle_color_coef) * self.o * self.o / self.h
        self.price_buy = np.max([self.price_color_white, self.price_level_up]) # цены открытия новой позиции вверх, стоп шорта
        self.price_sell = np.min([self.price_color_black, self.price_level_down]) # цены открытия новых позиции вниз, стоп лонга
        tr = self.trader
        if tr.position == 0:
            opened = tr.open_at_price(1, self.price_buy)
            if opened:
                print (" @open long")
            else:
                opened = tr.open_at_price(-1, self.price_sell)
                if opened:
                    print (' @open short')
        if tr.position != 0:
            if self.trader.position == 1:
                reversed = tr.reverse_at_price(self.price_sell)
                if reversed:
                    print (' @reverse long to short')
            else:
                if tr.position == -1:
                    reversed = tr.reverse_at_price(self.price_buy)
                    if reversed:
                        print (' @reverse short to long')

        #print ' price levels', self.price_level_up, self.price_level_down
        #print ' color levels', self.price_color_white, self.price_color_black
        #print ' buy sell', self.price_buy, self.price_sell
        #print ' current d', self.d
        if self.d > self.d_level_plus:
            #print ' up', self.row.hl
            #if self.position == 0:
                #self.position = 1
            pass
        if self.d < self.d_level_minus:
            #print ' down', self.row.hl
            #if self.position == 0:
                #self.position = -1
            pass
