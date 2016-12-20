
# coding: utf-8

# для совместимости с версией 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import scipy.optimize as opt
#import numba

#Свойство первой статистики
#замечание по знаменателю: если есть два последовательных одинаковых приращения d, то
#знаменатель равен нулю при любом d: d^2 + d^2 - 2 * (( d + d ) / 2) ^ 2
#причем в этом случае уровнем разладки вверх/вниз будет среднее значение
#это ещё одна причина по которой следует ограничивать значение знаменателя
class Statistics(object):
    def __init__(self):
        self.t_levels_range = np.around(np.arange(0.7, 0.9, 0.025), 4)
        #self.t_levels_range = np.array([0.9])
        self.min_down_value = 0.01 #ограничим значения знаменателя
        self.limit_denominator = True
        if not self.limit_denominator:
            print('no denominator limit!')
        # (дисперсии) нужна оценка в зависимости от волантильности
        # позже сюда будет назначаться оценка дисперсии по экспериментальным
        # данным
        print('stat::need estimation of min dispersion! now min_d = {}'.format(self.min_down_value))

    #i - точка в которой считаем статистику
    #расчет статистики не используя скользящие средние
    #calc_point - свеча для которой считаем разладку, индекс считается от нуля
    #window_size - окно расчета статистики, включая расчетную свечку, t в обозначениях АГ
    def calc_stat1_direct(self, candles, calc_point, window_size, d = np.nan, t_level = 0.75):
        if calc_point < 3:#не меньше чем четыре свечи
            raise Exception('start point less than 3!')
        if window_size < 3:
            raise Exception('window size less than 3!')
        if window_size > calc_point:
            raise Exception('windows size more than start point!')
        if np.isnan(d):
            d = candles.d.ix[calc_point]
        i = window_size
        fl_i = float(i)
        im1 = i - 1
        fl_im1 = float(im1)
        mean = candles.d.ix[calc_point - im1:calc_point].mean()
        sum_sq = candles.d_sq.ix[calc_point - im1:calc_point].sum()
        a = np.sqrt(fl_i - 2.) * np.sqrt(fl_im1 / fl_i)
        down = 0. + np.sqrt(sum_sq - fl_im1 * mean * mean)
        if self.limit_denominator:
            if down < self.min_down_value:
                down = self.min_down_value
        up = 0. + a * (d - mean)
        result = up  / down
        t_value = stat.t.ppf(t_level, i-2)
        b = down * t_value / a
        level_up = b + mean #считаем критические приращения
        level_down = - b + mean
        return result, np.abs(result) > t_value, [level_up, level_down]

    #функция расчета статистики разладки
    #в режиме fforecast оценивает разладку от прогнозного значения дельты
    #Solve[a * (d - e) / b == t, d]
    #замечание по знаменателю: если есть два последовательных одинаковых приращения d, то
    #знаменатель равен нулю при любом d: d^2 + d^2 - 2 * (( d + d ) / 2) ^ 2
    #@numba.jit
    def calc_stat1(self, candles, i, d = [], t_level_forecast = 0.75):
        # i ширина окна относительно текущего приращения
        # шир
        if i < 3:
            raise Exception('window size less than 3!')
        fforecast = np.size(d) > 0 #режим расчета прогноза
        fl_i = float(i)
        im1 = i - 1
        fl_im1 = float(im1)
        pref_forecast = 'f'
        mean_cn = 'mean{}'.format(im1) # среднее предыдущих приращений
        sum_sq_cn = 'sum_sq{}'.format(im1) # сумма квадратов предыдущих приращений
        col_up = 'stat1_up{}'.format(i)
        col_down = 'stat1_down{}'.format(i)
        col_dc_plus = 'dcr1_plus{}_'.format(i) #приращение при котором происходит разладка вверх
        col_dc_minus = 'dcr1_minus{}_'.format(i) #приращение при котором происходит разладка вниз
        if not fforecast:
            candles[pref_forecast + mean_cn] = pd.rolling_mean(candles.d, im1) #среднее для расчета прогноза включает текущую точку
            candles[mean_cn] = candles[pref_forecast + mean_cn].shift(1) #среднее по предыдущим точкам
            candles['d_sq'] = candles.d * candles.d
            candles[pref_forecast + sum_sq_cn] = pd.rolling_sum(candles.d_sq, im1)
            candles[sum_sq_cn] = candles[pref_forecast + sum_sq_cn].shift(1)
            current_d = candles.d
        else:
            current_d = d
            mean_cn = pref_forecast + mean_cn
            sum_sq_cn = pref_forecast + sum_sq_cn
        mean = candles[mean_cn]
        sum_sq = candles[sum_sq_cn]
        #print fl_i, fl_im1
        a = np.sqrt(fl_i - 2.) * np.sqrt(fl_im1 / fl_i)
        down = np.sqrt(sum_sq - fl_im1 * mean * mean)
        up = a * (current_d - mean)
        #print mean, sum_sq
        #print a, up, down
        if self.limit_denominator: # up numerator
            if not fforecast:
                down.fillna(0,  inplace=True)
                down[down < self.min_down_value] = self.min_down_value #орграничение снизу значения знаменателя (дисперсии)
            else:
                if down < self.min_down_value:
                    down = self.min_down_value
        result = up  / down
        if not fforecast:
            candles[col_down] = down
            candles[col_up] = up
            #candles.ix[candles[col_down] < self.min_down_value, col_down] = self.min_down_value #орграничение снизу значения знаменателя (дисперсии)
            #candles[col_down][candles[col_down] < self.min_down_value] = self.min_down_value
            candles['stat1_{}'.format(i)] = result
            #t критерий стьюдента для первой статистики с i-2 степенями свободы
            for t_level in self.t_levels_range:
                t_value = stat.t.ppf(t_level, i-2)
                candles['t_{}_{}'.format(i,t_level)] = t_value
                b = down * t_value / a
                candles[col_dc_plus + '{}'.format(t_level)] = b + mean #считаем критические приращения
                candles[col_dc_minus + '{}'.format(t_level)] = - b + mean
            f_level_up = 0
            f_level_down = 0
        else:
            t_value = candles['t_{}_{}'.format(i, t_level_forecast)]
            b = down * t_value / a
            f_level_up = b + mean # при какой величине прогноза произойдет разладка вверх
            f_level_down = - b + mean #... разладка вниз
        return result, np.abs(result) > t_value, [f_level_up, f_level_down]

    #функция расчета статистики подтверждения разладки
    #Solve[a * ((d + f) / 2 - e) / b == t, d]
    def calc_stat2_1(self, candles, i):#i ширина окна относительно текущего приращения
        if i < 4:
            raise Exception('window size less than 4!')
        fl_i = tp1 = float(i) # t + 1, t - обозначение из видео АГ, индекс шага когда произошла разладка
        im1 = i - 1 # t
        fl_im1 = t = float(im1)
        im2 = i - 2 # t-1
        fl_im2 = tm1 = float(im2)
        mean_cn = 'mean{}'.format(im2) #колонка со средним
        sum_sq_cn = 'sum_sq{}'.format(im2) #колонка с суммой квадратов
        suffix = '_1'
        col_dc_plus = 'dcr2_plus{}_'.format(i) #приращение при котором происходит разладка вверх
        col_dc_minus = 'dcr2_minus{}_'.format(i) #приращение при котором происходит разладка вниз
        candles[mean_cn + suffix] = candles[mean_cn].shift(1) # расчитано в первой статистике, просто сдвигаем
        candles[sum_sq_cn + suffix] = candles[sum_sq_cn].shift(1)
        mean_cn = mean_cn + suffix
        sum_sq_cn = sum_sq_cn + suffix
        preffix = 'stat2-1_'
        col_down = '{}down{}'.format(preffix, i)
        col_up = '{}up{}'.format(preffix, i)
        sum_sq = candles[sum_sq_cn]
        mean = candles[mean_cn]
        a = np.sqrt(tm1 - 1.) * np.sqrt(2 * tm1 / tp1)
        down = np.sqrt(sum_sq - tm1 * mean * mean)
        up = a * ((candles.d + candles.d1) / 2 - mean)
        candles['{}{}'.format(preffix, i)] = up  / down
        candles[col_up] = up
        candles[col_down] = down
        #candles['{}{}sh1'.format(preffix, i)] = candles['{}{}'.format(preffix, i)].shift(-1)
        #t критерий стьюдента для второй статистики с t-2 степенями свободы
        for t_level in self.t_levels_range:
            t_value = stat.t.ppf(t_level, tm1 - 1) # t-2 степеней свободы
            candles['t2-1_{}_{}'.format(i,t_level)] = t_level
            b = down * t_value / a
            candles[col_dc_plus + '{}'.format(t_level)] = 2 *(b + mean) - candles.d1 #считаем критические приращения
            candles[col_dc_minus + '{}'.format(t_level)] = 2 * (- b + mean) - candles.d1
            #pass

class DisorderSeacher:
    def __init__(self, calculator):
        self.candles = None
        self.i = 0
        self.t_value = 0
        self.t_level = 0
        self.calculator = calculator
        self.startValue = 100

    def stat_value(self, d):
        st_value, disorder = self.calculator.calc_stat1(self.candles, self.i, d, self.t_level)
        return st_value

    def eqPlus(self, d):
        return self.stat_value(d) - self.t_value

    def eqMinus(self, d):
        return self.stat_value(d) + self.t_value

    def find_disorder_d(self, row, i, t_level):#ищем приращение при котором происходит разладка
        self.candles = row
        self.i = i
        self.t_value = stat.t.ppf(t_level, i-2)
        self.t_level = t_level
        root1, infodict, ier, mesg = opt.fsolve(self.eqPlus, self.startValue, full_output = True)
        #print infodict
        if not ier == 1:
            #raise Exception(mesg)
            #print mesg
            pass
        if root1 == self.startValue:
            raise Exception('root is start value')
        root2, infodict, ier, mesg = opt.fsolve(self.eqMinus, -self.startValue, full_output = True)
        if not ier == 1:
            #raise Exception(mesg)
            pass
        if root1 == -self.startValue:
            raise Exception('root is start value')
        #print infodict, ier
        return root1, root2

class Disorder(object):
    def __init__(self):
        self.clear()
        self.col_stat1 = 'stat1_{}'
        self.col_stat2 = 'stat2-1_{}'
        self.col_t1 = 't_{}_{}'
        self.col_t2 = 't2-1_{}_{}'
        self.t_level1 = 0.75
        self.t_level2 = 0.75

    def clear(self):
        """

        :rtype: object
        """
        self.disorder1 = False
        self.disorder2 = False
        self.step = 0
        self.sign1 = 0
        self.sign2 = 0
        self.color1 = 0
        self.color2 = 0

    def stat1value(self, row, windowSize):
        return row[self.col_stat1.format(windowSize)]

    def t1value(self, row, windowSize):
        return row[self.col_t1.format(windowSize, self.t_level1)]

    def stat2value(self, row, windowSize):
        return row[self.col_stat2.format(windowSize)]

    def t2value(self, row, windowSize):
        return row[self.col_t2.format(windowSize, self.t_level2)]

    #@numba.jit
    def Check(self, row, windowSize):
        if self.step == 0:
            self.disorder1 = (np.abs(self.stat1value(row, windowSize)) > self.t1value(row, windowSize))
            #print self.stat1value(row, windowSize), self.t1value(row, windowSize)
            if self.disorder1:
                self.sign1 = np.sign(self.stat1value(row, windowSize))
                self.step = 1
                self.color1 = row.color
        elif self.step == 1:
            if not self.disorder1:
                raise Exception('no first disorder!')
            self.disorder2 =  np.abs(self.stat2value(row, windowSize)) > self.t2value(row, windowSize)
            self.sign2 = np.sign(self.stat2value(row, windowSize))
            if not self.disorder2 or not self.sign2 == self.sign1:
                self.disorder2 = False
                self.sign2 = 0
            self.step = 2
            self.color2 = row.color
        elif self.step == 2:
            raise Exception('cheking stat at step 2!')
