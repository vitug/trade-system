# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import numba
import gorchakov
import stocks
import traders
import intraday
import datetime


class OptimizerBase(object):
    def __init__(self):
        self.parameters = np.array([])
        self.equity_long = [] #np.array([])
        self.equity_short = [] #np.array([])

    def test(self):
        pass

    def prepare(self):
        pass

    def do(self):
        pass

class Optimizer(OptimizerBase):
    def __init__(self, asset):
        OptimizerBase.__init__(self)
        self.asset = asset
        #self.windowSize = 3
        #self.alpha_range = np.arange(.55, 0.95, 0.025)
        self.alpha_range = asset.statCalculator.t_levels_range
        #self.beta_range = [.5]
        self.t_level = 0
        self.use_forecast = True

    def test(self):
        #disorder.t_level1 = 0.75; disorder.t_level2 = .5;
        i = 0
        candles = self.asset.candles
        disorder = self.disorder
        trader = self.trader
        asset = self.asset
        windowSize = 3
        self.disorder.t_level2 = self.disorder.t_level1 = self.t_level
        candles['disorders2'] = 50.
        candles['disorders1'] = 50.
        candles['long_position'] = 0.
        candles['short_position'] = 0.
        candles['long_position_d'] = np.nan
        candles['short_position_d'] = np.nan
        candles['long_position_open'] = np.nan
        candles['long_position_close'] = np.nan
        candles['short_position_open'] = np.nan
        candles['short_position_close'] = np.nan
        candles['trend'] = 50
        tradePosition = 0
        disorder.clear()
        trader.Position = 0
        #print '?forecast row color'
        for index, row in candles.iterrows():
            if i < 3:#пропускаем первые значения для которых приращения и средние неопределены
                i += 1
                continue

            disorder.Check(row, windowSize)

            if disorder.step == 1 and disorder.disorder1:
                #print index
                #print ' first disorder ', windowSize, disorder.sign1, row.color, row.d
                candles.set_value(index,'disorders1',row.hl)

            if disorder.step == 2 and disorder.disorder2:
                #print ' ++second disorder', windowSize
                windowSize = 2
                candles.set_value(index,'disorders2',row.hl)

            if disorder.step == 2 and not disorder.disorder2:
                #print ' --first disorder not approved'
                pass

            #forecast = asset.forecast_coef_true * row.chl
            #forecast = asset.forecast_true_coef * row.chl + asset.forecast_true_error
            #print asset.forecast_coef * row.chl + asset.forecast_error
            forecast = np.array([row.fore_low, row.fore_high])
            #print forecast
            forecast_stat, forecast_disorder, levels = asset.statCalculator.calc_stat1(row, windowSize + 1, forecast, disorder.t_level1)
            if forecast_disorder.all():
                #print ' @forecast disorder', forecast_stat, forecast_disorder, forecast
                pass

            use_forecast = self.use_forecast

            forecast_long_condition = (use_forecast and forecast_disorder.all() and (forecast_stat > 0).all() and row.color > 0)
            forecast_short_condition = (use_forecast and forecast_disorder.all() and (forecast_stat < 0).all() and row.color < 0)

            #forecast_long_condition = (use_forecast and forecast_disorder.all() and (forecast_stat > 0).all())
            #forecast_short_condition = (use_forecast and forecast_disorder.all() and (forecast_stat < 0).all())

            #торговля #могли не открыть позицию на первой разладке
            longCondition = (disorder.step == 1 and disorder.disorder1 and disorder.sign1 > 0 and row.color > 0) or \
                            (disorder.step == 2 and not disorder.disorder2 and disorder.sign1 < 0 and disorder.color1 < 0) or \
                            (disorder.step == 2 and disorder.disorder2 and disorder.sign1 > 0 and row.color > 0) or \
                            forecast_long_condition

            shortCondition = (disorder.step == 1 and disorder.disorder1 and disorder.sign1 < 0 and row.color < 0) or \
                            (disorder.step == 2 and not disorder.disorder2 and disorder.sign1 > 0 and disorder.color1 > 0) or \
                            (disorder.step == 2 and disorder.disorder2 and disorder.sign1 < 0 and row.color < 0) or \
                            forecast_short_condition

            #print " ##", longCondition, shortCondition

            if longCondition and shortCondition:
                #raise Exception('long and short')
                longCondition = forecast_long_condition
                shortCondition = forecast_short_condition

            if longCondition:
                tradePosition = 1
            if shortCondition:
                tradePosition = -1

            if tradePosition == 1:
                if not np.isnan(row.dp1):
                    candles.set_value(index,'long_position',row.dp1)
                    candles.set_value(index,'short_position',0)

            if tradePosition == -1:
                if not np.isnan(row.dp1):
                    candles.set_value(index,'long_position',0)
                    candles.set_value(index,'short_position',row.dp1)

            #торгуем
            trader.Trade(tradePosition,candles,row,index)
            #сбрасываем разладки в исходное состояние на шаге 2
            if disorder.step == 2:
                disorder.clear()

            windowSize += 1
            i += 1

        candles['short_position_d'] = -np.log(candles.short_position_close / candles.short_position_open).fillna(0).shift(-1)
        candles['long_position_d'] = np.log(candles.long_position_close / candles.long_position_open).fillna(0).shift(-1)
        #candles.long_position.cumsum().plot()

        plt.plot(candles.long_position_d.cumsum(), label = 'long {}'.format(self.disorder.t_level1))
        #plt.plot(candles.short_position_d.cumsum(), label = 'short {}'.format(self.disorder.t_level1))
        plt.draw()
        self.parameters = np.append(self.parameters, self.t_level)
        self.equity_long.append(candles.long_position_d.copy())
        self.equity_short.append(candles.short_position_d.copy())

    def prepare(self):
        self.disorder = gorchakov.Disorder()
        self.trader = traders.Trader()
        #self.parameters['t_level'] = 0

    def do(self):
        self.prepare()
        #self.figure = plt.figure(figsize=(10, 65))
        self.graph_col = 2
        self.graph_row = np.round(float(len(self.alpha_range)) / self.graph_col) + 1
        i = 0
        start_time = datetime.datetime.now()
        for t_level in self.alpha_range:
            self.t_level = t_level
            #sp =  self.figure.add_subplot(self.graph_row, self.graph_col, i + 1 )
            #handles, labels = plt.ax.get_legend_handles_labels()
            #plt.legend()
            print (t_level)

            #self.test(); raise Exception('debug line')

            try:
                self.test()
            except:
                print ('error in test!')
            i += 1
        end_time = datetime.datetime.now()
        calculation_time = (end_time - start_time)
        print ('calc time', calculation_time)
        leg = plt.legend(bbox_to_anchor=(.05, 1), loc=2, borderaxespad=0.)
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)
        ax = plt.gca()
        #ax2 = plt.twinx(ax)
        plt.plot(self.asset.minutes.d.cumsum(),'--')
        #ax2.plot(self.asset.candles.c,'o--')
        #ax2.plot(self.asset.candles.l,'o--')

#numpy
class OptimizerIntraday(Optimizer):

    def test0(self):
        #optimization system
        #disorder.t_level1 = 0.75; disorder.t_level2 = .8;
        columns = [u'o', u'h', u'l', u'c', u'v', u'hl', u'd', u'i']
        compare = columns == self.asset.minutes.columns
        if not compare.all():
            raise Exception('Columns are different!')
        self.intraday = intraday.Intraday(self.t_level, self.asset)
        i = 0
        for index, row in self.asset.candles.iterrows():
            if i < 4:#пропускаем первые значения для которых приращения и средние неопределены
                i += 1
                continue
            intra_minutes = self.asset.minutes[row.beginTime:row.nextTime]
            intra_matrix = intra_minutes.as_matrix()
            self.intraday.log.info(index)
            self.intraday.init(row, intra_matrix)
            for intra_row in intra_matrix:
                self.intraday.check_candle(intra_row)
        self.intraday.trader.calc()
        self.intraday.trader.equity_plot(log = True, label_suffix='{}'.format(self.t_level))
        #self.intraday.log.close()
        self.parameters = np.append(self.parameters, self.t_level)
        self.equity_long.append(self.intraday.trader.candles.l_log_d.copy())
        self.equity_short.append(-self.intraday.trader.candles.sh_log_d.copy())

    #@numba.jit
    def test(self):
        #эта функция работает в два раза быстрее чем test0, функция check_candle вызывается только на экстремумах
        #и в конце периода
        #intraday trading
        #optimization system
        #disorder.t_level1 = 0.75; disorder.t_level2 = .8;
        columns = [u'o', u'h', u'l', u'c', u'v', u'hl', u'd', u'i']
        compare = columns == self.asset.minutes.columns
        if not compare.all():
            raise Exception('Columns are different!')
        self.intraday = intraday.Intraday(self.t_level, self.asset)
        i = 0
        for index, row in self.asset.candles.iterrows():
            if i < 4:#пропускаем первые значения для которых приращения и средние неопределены
                i += 1
                continue
            intra_minutes = self.asset.minutes[row.beginTime:row.nextTime]
            intra_matrix = intra_minutes.as_matrix()
            self.intraday.log.info(index)
            self.intraday.init(row, intra_matrix)
            ih = 1
            il = 2
            max = intra_matrix[0, ih]
            min = intra_matrix[0, il]
            i = 0
            new_extremum = True
            matrix_length = intra_matrix.shape[0]
            for intra_row in intra_matrix:
                if intra_row[ih] > max:
                    new_extremum = True
                    max = intra_row[ih]
                if intra_row[il] < min:
                    new_extremum = True
                    min = intra_row[il]
                if new_extremum or i == matrix_length - 1:
                    self.intraday.check_candle(intra_row)
                new_extremum = False
                i += 1
        self.intraday.trader.calc()
        self.intraday.trader.equity_plot(log = True, label_suffix='{:.3f}'.format(self.t_level))
        #self.intraday.log.close()
        self.parameters = np.append(self.parameters, self.t_level)
        self.equity_long.append(self.intraday.trader.candles.l_log_d.copy())
        self.equity_short.append(-self.intraday.trader.candles.sh_log_d.copy())

    def prepare(self):
        pass
        #self.intraday = intraday.Intraday(self.t_level, self.asset)

#pandas
#работает медленно из-за переиндексации таблиц
class OptimizerIntraday_pandas(Optimizer):

    def test(self):
        #print 'test'
        #return
        #intraday trading
        #что требуется
        #прогноз на сегодняшнюю разладку, прогноз цвета свечи, прогноз на завтрашнюю разладку
        #optimization system
        #disorder.t_level1 = 0.75; disorder.t_level2 = .8;
        columns = [u'o', u'h', u'l', u'c', u'v', u'hl', u'i']
        compare = columns == self.asset.minutes.columns
        if not compare.all():
            raise 'Columns are different!'
        self.intraday = intraday.Intraday_pandas(self.t_level, self.asset)
        i = 0
        for index, row in self.asset.candles.iterrows():
            if i < 4:#пропускаем первые значения для которых приращения и средние неопределены
                i += 1
                prev_row = row
                continue
            intra_minutes = self.asset.minutes[row.beginTime:row.nextTime]
            self.intraday.log.info(index)
            self.intraday.init(row, intra_minutes)
            for intra_index, intra_row in intra_minutes.iterrows():
                self.intraday.check_candle(intra_row)
        self.intraday.trader.calc()
        self.intraday.trader.equity_plot(log = True, label_suffix='{:.3f}'.format(self.t_level))
        self.intraday.log.close()

    def prepare(self):
        pass
        #self.intraday = intraday.Intraday(self.t_level, self.asset)

