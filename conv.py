"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time

import numpy
import numpy as np 

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression 
from mlp import HiddenLayer

from cifar import load_data

rng = numpy.random.RandomState(23455)

class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

def combine(xs, ys):
  assert isinstance(xs, ArrayList) or isinstance(ys, ArrayList)
  if not isinstance(ys, ArrayList):
    ys = ArrayList([ys] * len(xs))
  elif not isinstance(xs, ArrayList):
    xs = ArrayList([xs] * len(ys))
  return xs.zip(ys)

class ArrayList(object):
  def __init__(self, arrays = []):
    self.arrays = arrays

  def __str__(self):
    return "ArrayList(%s)" % ("; ".join(str(array.shape) for array in self.arrays))

  def __len__(self):
    return len(self.arrays)

  def zip(self, other):
    if np.isscalar(other):
      other = [other] * len(self)
    if isinstance(other, list):
      other = ArrayList(other)
    assert isinstance(other, ArrayList)
    assert len(other) == len(self)
    return ArrayList(zip(self.arrays, other.arrays))
    
  def __iter__(self):
    return iter(self.arrays)

  def dot(self, other):
    return sum((x*y).sum() for x,y in combine(self,other))

  def norm(self):
    return np.sqrt(self.dot(self))

  def __neg__(self):
    return ArrayList([-x for x in self])

  def __add__(self, other):
    return ArrayList([x+y for x,y in combine(self, other)])

  def __sub__(self, other):
    return ArrayList([x-y for x,y in combine(self, other)])

  
  def __mul__(self, other):
    return ArrayList([x*y for x, y in combine(self, other)])
 
  def __rmul__(self, other):
    return self * other

  def __div__(self, other):
    return ArrayList([x/y for x, y in combine(self, other)])

class ArrayListAccumulator(object):
  def __init__(self):
    self.arrays = []
    self.n_updates = 0

  def add(self, xs):
    self.n_updates += 1
    if not isinstance(xs, ArrayList):
      xs = ArrayList(xs)
    if len(self.arrays) == 0:  
      self.arrays = xs
    else:
      self.arrays = self.arrays + xs

  def sums(self):
    return self.arrays
  
  def averages(self):
    return self.arrays / self.n_updates 

def mean(xss):
  acc = ArrayListAccumulator()
  for xs in xss:
    acc.add(xs)
  return acc.averages()

def weighted_mean(xss, weights):
  acc = Accumulator()
  for xs, w in zip(xss, weights):
    acc.add(mult(xs, w))
  return acc.sums() / np.sum(weights)

class Network(object): 
  def __init__(self,  
                     mini_batch_size, 
                     learning_rate, 
                     momentum, 
                     n_filters, 
                     n_out = 10,
                     input_height = 32, 
                     input_width = 32, 
                     n_colors = 3):
   
    self.mini_batch_size = mini_batch_size  
    self.momentum = momentum 
    self.learning_rate = learning_rate
    # allocate symbolic variables for the data
    x = T.tensor4('x') # matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '  >> Building model: mini_batch_size = %d, learning_rate = %s, momentum = %s, n_filters = %s' % (mini_batch_size, learning_rate, momentum, n_filters)

    # Reshape matrix of rasterized images 
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((mini_batch_size, n_colors, input_height, input_width))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    pool_size = (2,2)
    filter_size = (5,5) 
    layer0 = ConvPoolLayer(rng, input=layer0_input,
            image_shape=(mini_batch_size, n_colors, input_height, input_width),
            filter_shape=(n_filters[0], n_colors, filter_size[0], filter_size[1]), 
            poolsize=pool_size)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = ConvPoolLayer(rng, input=layer0.output,
            image_shape=(mini_batch_size, n_filters[0], 14, 14),
            filter_shape=(n_filters[1], n_filters[0], filter_size[0], filter_size[1]), 
            poolsize=pool_size)

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = \
      HiddenLayer(rng, input=layer2_input, n_in=n_filters[1] * filter_size[0] * filter_size[1],
                  n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=n_out)

    # the cost we minimize during training is the NLL of the model
    self.cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    self.test_model = theano.function([x,y], layer3.errors(y)) 
    self.params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    self.grads = T.grad(self.cost, self.params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(self.params, self.grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    # self.train_model = theano.function([x,y], self.cost)# , updates=updates)
    # self.fprop = theano.function([x,y], self.cost)
    self.bprop_grads = theano.function([x, y], self.grads)

    self.bprop_update = theano.function([x, y], self.cost, updates = updates)

  def get_weights_list(self):
    return [p.get_value(borrow=True) for p in self.params]

  def get_weights(self):
    return ArrayList(self.get_weights_list())
 
  def get_weights_array(self):
    weights_list = self.get_weights_list()
    return self.flatten(self.get_weights_list())

  def flatten(self,arrays): 
    elts_per_subarray = []
    is_array = []
    elt_type = None 
    for w in arrays:
      if isinstance(w, np.ndarray):
        elts_per_subarray.append(w.size)
        is_array.append(True)
        if elt_type is None:
          elt_type = w.dtype 
      else:
        assert np.isscalar(w)
        elts_per_subarray.append(1)
        is_array.append(False)
    total_elts = sum(elts_per_subarray)
    result = np.empty(shape = (total_elts,), dtype=elt_type)
    curr_idx = 0
    for (nelts, value) in zip(elts_per_subarray, arrays):
      if np.rank(value) > 1:
        value = np.ravel(value)
      result[curr_idx:(curr_idx + nelts)] = value 
      curr_idx += nelts 
    return result
 
  def add_list_to_weights(self, dxs):
    """
    Given a ragged list of arrays, add each to network params
    """
    for p, dx in zip(self.params, dxs):
      old_value = p.get_value(borrow = True)
      old_value += dx 
      p.set_value(old_value, borrow=True)
   
  def add_array_to_weights(self, dxs):
   """
   Given a long weight vector, split it apart and each component to its layers' weights
   """
   curr_idx = 0
   for p in self.params:
     w = p.get_value(borrow=True)
     if isinstance(w, np.ndarray):
       nelts = w.size
       w_flat = w.ravel()
       # pray and hope the memory from ravel isn't fresh
       w_flat += dxs[curr_idx:curr_idx+nelts]
       assert w_flat.strides == dxs.strides
       assert w.ctypes.data == w_flat.ctypes.data 
       p.set_value(w, borrow=True)
     else:
       assert np.isscalar(w)
       nelts = 1 
       p.set_value(w + dxs[curr_idx])
     curr_idx += nelts 

  def local_update_step_with_momentum(self, grads, old_dxs):
    new_dxs = []
    for (g, old_dx) in zip(grads, old_dxs):
      new_dx = np.array(g) 
      new_dx *= -self.learning_rate 
      old_dx *= self.momentum 
      new_dx += old_dx 
      new_dxs.append(new_dx)
    self.add_list_to_weights(new_dxs)
    return new_dxs 

  def average_gradients(self, x, y):
    """
    Get the average gradient across multiple mini-batches
    """
    combined = None
    n_batches = x.shape[0] / self.mini_batch_size
    for batch_idx in xrange(n_batches):
      start = batch_idx*self.mini_batch_size
      stop = start + self.mini_batch_size
      xslice = x[start:stop]
      yslice = y[start:stop]
      g_list = [np.array(g_elt) for g_elt in self.bprop_grads(xslice, yslice)]
      g_flat = self.flatten(g_list)
      if combined is None:
        combined = g_flat
      else:
        combined += g_flat
    combined /= n_batches
    return combined 
  
  def get_state(self, x, y):
    return self.get_weights_array(), self.average_gradients(x,y)
       
  def update_batches(self, x, y):
    costs = []
    def fn(xslice, yslice):
      cost = self.bprop_update(xslice, yslice)
      costs.append(cost)
      # new_dxs = self.local_update_step(grads, dxs)
      #del dxs[0:len(dxs)]
      #dxs.extend(new_dxs)
    for_each_slice(x, y, fn, self.mini_batch_size)
    return np.mean(costs) / self.mini_batch_size 
    # print "  Mean batch cost: %0.3f" % np.mean(costs)
    # compute changes by -original + final 
    #return final_weights, final_grads, initial_weights, initial_grads 

 
def for_each_slice(x, y, fn, mini_batch_size):
  for mini_batch_idx in xrange(x.shape[0] / mini_batch_size):
    start = mini_batch_idx * mini_batch_size 
    stop = start + mini_batch_size 
    xslice = x[start:stop]
    yslice = y[start:stop]
    fn(xslice, yslice)

class DistLearner(object):
  def __init__(self, 
               n_workers = 1,
               n_epochs = 200, # how many passes over the data?
               n_out = 10, # how many outputs?  
               n_filters = [20, 50], # how many convolutions in the first two layers of the network?  
               global_learning_rate = 0.1,  # step size for big steps of combined gradients
               local_learning_rate = 0.01,  # step size on each worker
               global_momentum = 0.05,  # momentum of global updates
               local_momentum = 0.05,   # momentum on each worker 
               mini_batch_size = 20,    # how many gradients does a worker average per local step?
               n_local_steps = 2,       # how many mini-batch steps does a worker take?
               approx_local_change = False, # if approx then compare first mini-batch vs. last mini-batch 
               newton_method = 'memoryless-bfgs', # options = 'memoryless-bfgs', 'memoryless-bfgs-avg', 'svd', None
               gradient_average = 'mean', # 'mean', 'best', 'weighted'
               weight_average = 'mean'): # 'mean', 'best', 'weighted'
    self.n_workers = n_workers
    self.n_epochs = n_epochs
    self.n_filters = n_filters 
    self.global_learning_rate = global_learning_rate 
    self.local_learning_rate = local_learning_rate 
    self.global_momentum = global_momentum 
    self.local_momentum = local_momentum 
    self.mini_batch_size = mini_batch_size
    self.n_local_steps = n_local_steps 
    self.approx_local_change = approx_local_change 
    self.newton_method = newton_method 
    self.gradient_average = gradient_average 
    self.weight_average = weight_average 

    self.nets = [Network(mini_batch_size = mini_batch_size, 
                    learning_rate = local_learning_rate, 
                    momentum = local_momentum, 
                    n_filters = n_filters, 
                    n_out = n_out)
                 for _ in xrange(n_workers)]

  def __str__(self):
    attrs = ["%s = %s" % (k,self.__dict__[k]) for k in self.sorted_keys()]
    s = ", ".join(attrs)
    return "DistLearner(%s)" % s 
  
  def sorted_keys(self):
    return sorted(self.__dict__.keys())
  
  def sorted_values(self):
    return [self.__dict__[k] for k in self.sorted_keys()]
  
  def __eq__(self, other):
    ks1 = self.sorted_keys()
    ks2 = self.sorted_keys()
    if len(ks1) != len(ks2):
      return False
    for (k1,k2) in zip(ks1,ks2):
      if k1 != k2 or self.__dict__[k1] != other.__dict__[k2]:
        return False 
    return True
 
  def __hash__(self):
    return hash( (tuple(self.sorted_keys()), tuple(self.sorted_values())) )

  def fit(self, train_set_x, train_set_y, shuffle = False):
    ntrain, ncolors, image_rows, image_cols = train_set_x.shape
    # compute number of minibatches for training, validation and testing
    worker_batch_size = self.mini_batch_size * self.n_local_steps

    start_time = time.clock()
    for epoch in xrange(self.n_epochs):       
      print "  epoch", epoch 
      if shuffle:
        shuffle_indices = np.arange(ntrain)
        np.random.shuffle(shuffle_indices)
        train_set_x = train_set_x[shuffle_indices]
        train_set_y = train_set_y[shuffle_indices]
      start_idx = 0 
      last_print_idx = 0
      while ntrain - start_idx >= worker_batch_size * self.n_workers: 
          ws = []
          gs = []
          ss = []
          ys = []
          costs = []
          for worker_idx  in xrange(self.n_workers):
              batch_start = start_idx + worker_idx * worker_batch_size
              batch_stop = batch_start + worker_batch_size 

              batch_x = train_set_x[batch_start:batch_stop]
              batch_y = train_set_y[batch_start:batch_stop]
              net = self.nets[worker_idx]
              if self.newton_method is not  None:
                old_w, old_g = net.get_state(batch_x, batch_y) 
              costs = net.update_batches(batch_x, batch_y)
              w,g = net.get_state(batch_x, batch_y)
              ws.append(w)
              gs.append(g)
              if self.newton_method is not None:
                s = w - old_w 
                y = g - old_g 
                ss.append(s)
                ys.append(y)
          cost = np.mean(costs)
          if start_idx - last_print_idx >= 100:
            print "  Sample %d / %d, mean batch cost = %0.3f" % (start_idx, ntrain, cost) 
            last_print_idx = start_idx 
          if self.gradient_average == 'mean':
            g = np.mean(gs,axis=0)
          else:
            assert False, "Not implemented: gradient_average = %s" % self.gradient_average 
          if self.weight_average == 'mean':
            w = np.mean(ws,axis=0)
          else:
            assert False, "Not implemented: weight_average = %s" % self.weight_average 
 
          if self.newton_method == 'memoryless-bfgs':
              rhos = []
              alphas = []
              for (s,y) in zip(ss,ys):
                  rho = 1.0 / np.dot(s,y)
                  rhos.append(rho)
                  alpha = rho * np.dot(s,g)
                  alphas.append(alpha)
                  g -= alpha * y 
              for (i,(s,y)) in reversed(list(enumerate(zip(ss,ys)))):
                  alpha = alphas[i]
                  rho = rhos[i]
                  beta = rho * np.dot(y,g) 
                  g += s * (alpha - beta)
          elif self.newton_method == 'svd':
              Y = np.array(ys)
              S = np.array(ss)  
              U, D, V = np.linalg.svd(Y.T, full_matrices=False)
              diag_ratios = D / D[0] 
              cutoff = 0.0001 
              below_cutoff = diag_ratios < cutoff 
              if np.any(below_cutoff):
                k = np.argmax(below_cutoff)
              else:
                k = len(D)
              # print "  SVD rank = %d / %d (D = %s)" % (k, len(D), D) 
              k = np.argmax( (D / D[0]) < 10**-6) + 1
              U = U[:, :k]
              D = D[:k]
              V = V[:k, :]
              Ug = np.dot(U.T, g)
              DinvUg = np.dot(np.diag(1.0 / D), Ug)
              VDinvUg = np.dot(V.T, DinvUg)
              g = np.dot(S.T, VDinvUg)                
          else:
              assert self.newton_method is None, "Unrecognized newton method: %s" % self.newton_method 
          
          g *= -self.global_learning_rate  
          for worker_idx in xrange(self.n_workers):
            self.nets[worker_idx].add_array_to_weights(g)
          start_idx += worker_batch_size * self.n_workers 
    end_time = time.clock()
    elapsed = end_time - start_time 
    return elapsed 

  def score(self, test_set_x, test_set_y):
    """
    Return average accuracy on the test set
    """
    ntest = test_set_y.shape[0]
    errs = []
    def fn(xslice, yslice):
      errs.append(self.nets[0].test_model(xslice, yslice))
    for_each_slice(test_set_x, test_set_y, fn, self.mini_batch_size)
    return 1 - np.mean(errs)

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
  param_combos = all_combinations(
       n_workers = [1,8],
       mini_batch_size = [10, 50], 
       n_local_steps = [ 5, 20],  
       global_learning_rate = [5.0, 1.0, 0.1], # TODO: 'search'
       local_learning_rate = [0.1, 0.01, 0.001], # TODO: 'random'
       global_momentum = [0.0], # TODO: 0.05 
       local_momentum = [0.0], # TODO: 0.05 
       weight_average = ['mean'], # TODO: weighted, best
       gradient_average = ['mean'], # TODO: weighted, best 
       newton_method = [None, 'svd', 'memoryless-bfgs'])
  print "Generated %d parameter combinations" % len(param_combos)
  train_set_x, train_set_y, test_set_x, test_set_y  = \
    load_data(labels='coarse_labels')
  print "Train set:", train_set_x.shape
  print "Test set:", test_set_x.shape
  #train_set_x = train_set_x[:1000, :]
  #train_set_y = train_set_y[:1000]
  n_out = len(np.unique(test_set_y))
  n_epochs = 1
  best_acc = 0 
  best_acc_param = None
  best_acc_model = None 
  best_acc_time = None 


  best_acc_rate = 0
  best_acc_rate_param = None
  best_acc_rate_model = None 
  best_acc_rate_time = None 

  def print_best(i):
      print 
      print "=====" 
      print "After %d parameter combinations" % (i+1) 
      print
      print "Best w/ accuracy-rate %0.3f, training time = %s, model = %s" % (best_acc_rate*100.0, best_acc_rate_time, best_acc_rate_param)
      print
      print "Best  w/ accuracy %0.3f, training time = %s, model = %s" % (best_acc*100.0, best_acc_time, best_acc_param)
      print "====="
      print  
    
  for (i, params) in enumerate(param_combos):
    param_str = ", ".join("%s = %s" % (k,params[k]) for k in sorted(params))
    print "Param #%d" % (i+1), param_str 
    model = DistLearner(n_epochs = n_epochs, n_out = n_out, **params)

    elapsed_time = model.fit(train_set_x, train_set_y, shuffle = True)               
    acc = model.score(test_set_x, test_set_y)

    acc_rate = acc / elapsed_time 
    print "  Elapsed time: %0.3f seconds" % elapsed_time
    print "  Accuracy = %0.4f" %  (100.0 * acc)
    print "  Accuracy per second = %0.4f" % (100.0 * acc_rate)     
    print 

    if acc > best_acc: 
      best_acc = acc
      best_acc_model = model 
      best_acc_params = param_str
      best_acc_time = elapsed_time 
    if acc_rate > best_acc_rate:
      best_acc_rate = acc_rate 
      best_acc_rate_model = model
      best_acc_rate_param = param_str 
      best_acc_rate_time = elapsed_time 

    if i % 10 == 0:
      print_best(i)
  print
  print "DONE!" 
  print 
  print_best(i)
