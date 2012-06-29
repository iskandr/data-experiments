import numpy as np

def returns(v, lags, include_original = False, return_weight = 1):
  if not isinstance(lags, list):
    lags = [lags]
  # assuming data is 2d, so turn a 1d vector into an nx1 matrix
  if np.rank(v) == 1:
    v = np.reshape(v, (len(v), 1))
  
  n_lags = len(lags)

  n, d = v.shape
  max_lag = np.max(lags)
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

"""
TODO: Turn this into something reusable
In [89]: ytrain_return = np.log(train[200:, -1] / train[:-200, -1])

In [90]: ytest_return = np.log(test[200:, -1] / test[:-200, -1])

In [91]: ytest_return
Out[91]: 
array([  3.59749471e-05,   3.59749471e-05,   3.59749471e-05, ...,
        -2.64188592e-05,  -2.84154544e-05,  -2.84154544e-05])

In [92]: np.sum(np.abs(ytrain_return) > 0.0001)
Out[92]: 44631

In [93]: np.sum(np.abs(ytrain_return) > 0.00005)
Out[93]: 225815

In [94]: np.sum(np.abs(ytest_return) > 0.00005)
Out[94]: 35015

In [95]: ytrain_discrete = np.sign(ytrain_return) * (np.abs(ytrain_return) > 0.00005)

In [96]: ytest_discrete = np.sign(ytest_return) * (np.abs(ytest_return) > 0.00005)

In [97]: xtrain = train[:-200, :]

In [98]: xtest = test[:-200, :]

"""
