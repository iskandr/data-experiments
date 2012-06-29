import numpy as np

def returns(v, lags, include_original = False, return_weight = 1):
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
    returns = np.log(v[lag:, ] / v[:-lag, :]) 
    # skip beginning of returns series if it's from a lag shorter
    # than max_lag 
    result[:, i*d:(i+1)*d] = returns[(max_lag - lag):, :] * return_weight
  return result
  
def multiday_returns(v, lags, include_original = False, return_weight = 1, transpose_day = False):
  assert np.rank(v) == 3
  n_days = v.shape[0]
  result = []
  for i in xrange(n_days):
    day = v[i, :, :]
    if transpose_day:
      day = day.T
    result.append(returns(day, lags, include_original, return_weight))
  return np.vstack(result)
  
def split_data(v, lags, include_original=False, return_weight=1, transpose_day = False, test_start_day=2):
  assert np.rank(v) == 3
  n_days = v.shape[0]
  assert test_start_day < n_days
  train = v[:test_start_day, :, :]
  test = v[test_start_day:, :, :]
  train_data = multiday_returns(train, lags, include_original, return_weight, transpose_day)
  test_data = multiday_returns(test, lags, include_original, return_weight, transpose_day)
  return train_data, test_data 

def generate_dataset(v, lags, future_lag, target_idx, include_original=False, return_weight=1, transpose_day = False, test_start_day = 2):
  
  train, test = split_data(v, lags, \
    include_original=True, 
    return_weight=return_weight, 
    transpose_day = transpose_day, 
    test_start_day = test_start_day) 
  print train.shape[1]
  print len(lags)
  n_original_series = train.shape[1] / (len(lags)+1)
  print n_original_series
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
  
