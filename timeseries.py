import numpy as np

import pandas

  
def returns(v, lags, average_future_returns = True, include_original = False, return_weight = 1):
  if not isinstance(lags, list):
    lags = [lags]
  # assuming data is 2d, so turn a 1d vector into an nx1 matrix
  if np.rank(v) == 1:
    v = np.reshape(v, (len(v), 1))
  
  n_lags = len(lags)

  n, d = v.shape
  if len(lags) > 0:
    max_lag = np.max(lags)
  else: 
    max_lag = 0
  # have to truncate all the series to the same length 
  n_short = n - max_lag 
  if n_short <= 0:
    raise RuntimeError('time series too short: length = %d, max_lag = %d' % (n, max_lag))
  if include_original:
    result = np.zeros( (n_short, d*(n_lags+ 1)), dtype='float')
    result[:, d*n_lags : d*(n_lags+1) ] = v[max_lag:, :]
  else: 
    result = np.zeros( (n_short, d * n_lags), dtype='float')
  for i, lag in enumerate(lags):
    if average_future_returns:
      averaged = pandas.rolling_mean(v, lag)[lag:]
      returns = np.log(averaged / v[:-lag, :]) 
    else:
      returns = np.log(v[lag:, ] / v[:-lag, :]) 
    # skip beginning of returns series if it's from a lag shorter
    # than max_lag 
    result[:, i*d:(i+1)*d] = returns[(max_lag - lag):, :] * return_weight
  return result
  
def multiday_returns(v, lags, average_future_returns = False, include_original = False, include_time = False, return_weight = 1, transpose_day = False):
  assert np.rank(v) == 3
  n_days = v.shape[0]
  result = []
  for i in xrange(n_days):
    day = v[i, :, :]
    if transpose_day:
      day = day.T
    day_returns = returns(day, lags, average_future_returns, include_original, return_weight)
    
    if include_time:
      n_ticks = day_returns.shape[0]
      
      time = np.arange(n_ticks) / float(n_ticks)
      time_col = np.array([time]).T

      day_returns = np.hstack([ time_col, day_returns])
    result.append(day_returns)
  return np.vstack(result)
  
def split_data(v, lags, average_future_returns = False, include_original=False, include_time = False, return_weight=1, transpose_day = False, test_start_day=2):
  assert np.rank(v) == 3
  n_days = v.shape[0]
  assert test_start_day < n_days
  train = v[:test_start_day, :, :]
  test = v[test_start_day:, :, :]
  train_data = multiday_returns(train, lags, average_future_returns, include_original, include_time, return_weight, transpose_day)
  test_data = multiday_returns(test, lags, average_future_returns, include_original, include_time, return_weight, transpose_day)
  return train_data, test_data 

def generate_dataset(v, lags, future_lag, target_idx = 1, average_future_returns = True, include_original=False, include_time = True, return_weight=10000, transpose_day = True, test_start_day = 11):
  
  train, test = split_data(v, lags, \
    include_original=True, 
    average_future_returns = False, 
    include_time = include_time, 
    return_weight=return_weight, 
    transpose_day = transpose_day, 
    test_start_day = test_start_day) 

  n_cols = train.shape[1]
  n_original_series = (n_cols - (1 if include_time else 0)) / (len(lags)+1)
  
  
  # TODO: This is terrible, fix it! 
  if average_future_returns:
    train2, test2 = split_data(v, lags, \
      include_original=True, 
      average_future_returns = True, 
      include_time = include_time, 
      return_weight=return_weight, 
      transpose_day = transpose_day, 
      test_start_day = test_start_day) 
      
    train_target = train2[:, -(n_original_series - target_idx)]
    test_target = test2[:, -(n_original_series - target_idx)]
  else:
    train_target = train[:, -(n_original_series - target_idx)]
    test_target = test[:, -(n_original_series - target_idx)]
    
  ytrain_return = np.log(train_target[future_lag:] / train_target[:-future_lag])
  ytest_return = np.log(test_target[future_lag:] / test_target[:-future_lag])
  
  if not include_original:
    train = train[:, :-n_original_series]
    test = test[:, :-n_original_series]
  
  train = train[:-future_lag, :]
  test = test[:-future_lag, :]
  return train, ytrain_return, test, ytest_return
  

from regularized import RegularizedGaussianNB
from sklearn.linear_model import SGDClassifier

# search for best triplet of lags and best target_idx   
def search_for_best_dataset_params(v, possible_lags,  future_lags, \
    target_indices = [0], 
    return_weight = 10000, 
    transpose_day = True, 
    test_start_day = 2, 
    n_iter = 2):
  best_acc = 0
  best_params = None
  best_model = None
  target_index_accs = dict( [ (idx, []) for idx in target_indices])
  for l1 in possible_lags:
    for l2 in [l for l in possible_lags if l > l1]:
      for l3 in [l for l in possible_lags if l > l2]:
        lags = [l1, l2, l3]
        for target_idx in target_indices:
          for future_lag in [l for l in future_lags if l <= max(lags)]:
            print "Generating data for lags = %s, target_idx = %s, future_lag = %s" % (lags, target_idx, future_lag)
            
            xtrain, ytrain, xtest, ytest = generate_dataset(v, lags, \
              future_lag, target_idx, include_original = False, \
              return_weight = return_weight, 
              transpose_day = transpose_day, 
              test_start_day = test_start_day)
            
            lr = SGDClassifier(loss = 'log', n_iter = n_iter)
            print "Training classifier..." 
            lr.fit(xtrain, np.sign(ytrain))
            acc = lr.score(xtest, np.sign(ytest))
            target_index_accs[target_idx].append(acc)
            print "Accuracy", acc 
            if acc > best_acc:
              best_acc = acc
              best_params = {'lags': lags, 'target_idx': target_idx, 'future_lag': future_lag}
              best_model = lr 
  print "Best accuracy: %s, best_params: %s" % (best_acc, best_params)
  def stats(elts):
    return {'min': min(elts), 'max': max(elts), 'median':np.median(elts), 'mean':np.mean(elts), 'std': np.std(elts)}
  target_index_stats = dict([ (idx, stats(elts)) for (idx,elts) in target_index_accs.items()])
  print target_index_stats 
  return best_acc, best_params, best_model, target_index_stats
