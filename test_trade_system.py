# coding: utf-8
import stocks
import gorchakov
import intraday
import optimizer
import cProfile, pstats, StringIO
#import pickle
import cPickle as pickle

pr = cProfile.Profile()
profile = False

use_cache = True
if not use_cache:
    asset = stocks.Asset()
    #asset.load('data\sber14.txt')
    asset.load('data\micex14.txt')
    #asset.generate()

    asset.minutes = asset.minutes.between_time('10:00','18:40')

    asset.resample()
    #asset.report()
    pickle.dump( asset, open( "asset.dump", "wb" ) )
else:
    asset = pickle.load( open( "asset.dump", "rb" ) )

if profile:
    pr.enable()

print 'start...'
iOpt = optimizer.OptimizerIntraday(asset)
iOpt.do()
print 'end'

if profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

stocks.beep()