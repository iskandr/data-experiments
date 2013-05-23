import cPickle
import sys
import time

import numpy
import numpy as np 

from striate import ConvNet 

from cifar import load_data 
from dist_conv import DistConvNet 

def all_combinations(**params):
  combos = [{}]
  for (k,vs) in params.iteritems():
    if not isinstance(vs, (tuple, list)):
      vs = [vs]
    new_combos = []
    for v in vs:
      for old_params in combos:
        new_params = dict(old_params.iteritems())
        new_params[k] = v
        new_combos.append(new_params)
    combos = new_combos
  return combos 
   

from collections import namedtuple 
if __name__ == '__main__':
  n_epochs = 20
  posttrain_epochs = 5
  param_combos = all_combinations(
       n_workers = [2,1], 
       mini_batch_size = [64, 128], 
       n_local_steps = [ 20 ],  
       global_learning_rate = ['search', 1.0], # global_learning_rates,  # [0.1, 1.0, 2.0], # TODO: 'search'
       local_learning_rate = [0.01], # TODO: 'random'
       global_momentum = [0.0], # TODO: 0.05 
       local_momentum = [0.0], # TODO: 0.05 
       weight_average = ['best', 'weighted'], 
       gradient_average = ['best', 'weighted'],  
       newton_method = ['svd', 'memoryless-bfgs', None ], 
       conv_activation = ['relu', 'tanh'],
  ) 
  print "Generated %d parameter combinations" % len(param_combos)
  train_set_x, train_set_y, test_set_x, test_set_y  = \
    load_data(labels='coarse_labels')
  print "Train set:", train_set_x.shape
  print "Test set:", test_set_x.shape
  n_out = len(np.unique(test_set_y))
  best_acc = 0 
  best_acc_param = None
  best_acc_model = None 
  best_acc_time = None 
  results = {}
  accs = []

  def print_best(i):
      print 
      print "=====" 
      print "After %d parameter combinations" % (i+1) 
      print
      print "Accuracies: min %0.3f, max %0.3f, median %0.3f" % (np.min(accs), np.max(accs), np.median(accs))
      print 
      print "Best  w/ accuracy %0.3f, training time = %s, model = %s" % (best_acc*100.0, best_acc_time, best_acc_param)
      print "====="
      print  
 
  for (i, params) in enumerate(param_combos):
    param_str = ", ".join("%s = %s" % (k,params[k]) for k in sorted(params))
    print "Param #%d" % (i+1), param_str 
    model = DistConvNet(n_out = n_out, 
                       n_epochs = n_epochs, 
                       pretrain_epochs = 0, 
                       posttrain_epochs = posttrain_epochs,  **params)

    elapsed_time = model.fit(train_set_x, train_set_y, shuffle = False)               
    acc = model.score(test_set_x, test_set_y)
    accs.append(acc)
    results[param_str] = acc 

    baseline = 1.0 / n_out 
    acc_rate = (acc - baseline) / elapsed_time 
    print "  Elapsed time: %0.3f seconds" % elapsed_time
    print "  Accuracy = %0.4f" %  (100.0 * acc)
    print "  Accuracy per second = %0.4f" % (100.0 * acc_rate)     
    print 

    if acc > best_acc: 
      best_acc = acc
      best_acc_model = model 
      best_acc_param = param_str
      best_acc_time = elapsed_time 
    print_best(i)
  print
  print "DONE!" 
  print 
  print_best(i)
  results_file_name = 'results'
  f = open(results_file_name, 'w')
  cPickle.dump(results, f)
  f.close()
  keys = results.keys()
  values = results.values()
  sorted_indices = np.argsort(values) 
  for idx in sorted_indices:
    print "%0.3f %s" %   (values[i], keys[i])

