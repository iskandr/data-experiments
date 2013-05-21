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
from theano.misc.pycuda_utils import to_cudandarray, to_gpuarray 

from logistic_sgd import LogisticRegression 
from mlp import HiddenLayer

from cifar import load_data
import numexpr 
import pycuda 
import pycuda.autoinit 
from pycuda.gpuarray import GPUArray 

import scikits.cuda 
import scikits.cuda.linalg

scikits.cuda.linalg.init() 

import scikits.cuda.cublas as cublas 
cublas_handle =  cublas.cublasCreate()

rng = numpy.random.RandomState(23455)

class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), activation='relu'):
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
        if activation == 'tanh':
          self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
          assert activation == 'relu'
          self.output = T.maximum(0, pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) 

        # store parameters of this layer
        self.params = [self.W, self.b]

def gpu_copy(x):
  y = pycuda.gpuarray.empty_like(x)
  pycuda.driver.memcpy_dtod(y.gpudata, x.gpudata, x.nbytes)
  return y

def mean(gpu_arrays):
  acc = gpu_copy(gpu_arrays[0])
  n = len(gpu_arrays)
  if n == 1:
    return acc
  recip = 1.0 / n
  acc *= recip 
  for x in gpu_arrays[1:]:
    acc = acc.mul_add(1.0, x, recip)
  return acc 

def weighted_mean(gpu_arrays, weights):
  acc = gpu_arrays[0]
  weights /= np.sum(weights)
  acc *= weights[0]
  for x, w in zip(gpu_arrays[1:], weights[1:]):
    acc.mul_add(1.0, x, w) 
  return acc 

def memcpy(dest, src):
  assert len(dest.shape) == 1
  assert len(src.shape) == 1
  assert dest.nbytes == src.nbytes
  pycuda.driver.memcpy_dtod(dest.gpudata, src.gpudata, src.nbytes)

def concat(xs):
  if isinstance(xs[0], GPUArray):
    # Stupid GPUArray doesn't support strides
    # so have to go to stupid lengths to 
    # stack some gosh-darned vectors 
    elt_shape = xs[0].shape
    
    assert len(elt_shape) == 1
    elt_dtype = xs[0].dtype
    assert all(x.shape == elt_shape for x in xs)
    assert all(x.dtype == elt_dtype for x in xs)
    nrows = len(xs)
    row_nelts = elt_shape[0]
    total_nelts = nrows * elt_shape[0]
    result = pycuda.gpuarray.empty(shape=(total_nelts,), dtype=elt_dtype)
    for (i,x) in enumerate(xs):
      output_slice = result[i*row_nelts:(i+1)*row_nelts]
      memcpy(output_slice, x)
    final_shape = (nrows,) + elt_shape 
    return result.reshape(final_shape)
  else:
    return np.array(xs)

def scalar(x):
  if isinstance(x, GPUArray):
    return x.get().reshape(1)[0]
  else:
    assert np.isscalar(x)
    return x

def getidx(x, i):
  if isinstance(x, GPUArray):
    return x[i:i+1].get()[0]
  else:
    return x[i]

def dot(x,y):
  if isinstance(x, GPUArray):
    assert isinstance(y, GPUArray)
    if len(x.shape) == 1 and len(y.shape) == 1:
      return scalar(pycuda.gpuarray.dot(x,y))
    else:
      if len(x.shape) == 1:
        needs_ravel = True
        x = x.reshape((1,) + x.shape)
      if len(y.shape) == 1:
        needs_ravel = True
        y = y.reshape(y.shape + (1,))
      
      result = scikits.cuda.linalg.dot(x,y)
      if needs_ravel:
        assert result.shape[1] == 1 or result.shape[0] == 1
        result = result.ravel()
      return result 
  else:
    return np.dot(x,y)

def norm(x):
  if isinstance(x, GPUArray):
    return cublas.cublasSnrm2(cublas_handle, x.size, x.gpudata, 1)
  else:
    return np.linalg.norm(x) 

def diag_dot(diag, X):
  """
  Reweight the rows of X, as if multiplying on the left by a diagonal
  """
  d = len(diag)
  assert d <= X.shape[0]
  result = X[:d, :]    
  for row_idx in xrange(d): 
    row_slice = result[row_idx, :]
    row_slice *= getidx(diag, row_idx)
  return result 

def vecmin(x):
  if isinstance(x, GPUArray):
    return scalar(pycuda.gpuarray.min(x))
  else:
    return np.min(x)

def vecmax(x):
  if isinstance(x, GPUArray):
    return scalar(pycuda.gpuarray.max(x))
  else:
    return np.max(x)

def vecsum(x):
  if isinstance(x, GPUArray):
    return scalar(pycuda.gpuarray.sum(x))
  else:
    return np.sum(x)

def svd(X):
  if isinstance(X, GPUArray):
    return scikits.cuda.linalg.svd(X, jobu = 'S', jobvt='S')
  else:
    return np.linalg.svd(X, full_matrices=False)

def argmax(x):
  if isinstance(x, GPUArray):
    x = x.get()
  return np.argmax(x)

def transpose(X):
  if isinstance(X, GPUArray):
    return scikits.cuda.linalg.transpose(X)
  else:
    return X.T 

def take_rows(X, k):
  if isinstance(X, GPUArray):
    nrows, ncols = X.shape
    X_flat = X.ravel()
    assert k <= nrows
    dtype = X.dtype 
    result = pycuda.gpuarray.empty(shape = (k * ncols), dtype=dtype)
    input_slice = X_flat[:k*ncols]
    memcpy(result, input_slice)
    return result.reshape( (k, ncols) )
  else:
    return X[:k]
 
def take_cols(X, k):
  if isinstance(X, GPUArray):
    X = X.get()
    X = X[:, :k]
    return pycuda.gpuarray.to_gpu(X)
  else:
    return X[:, :k]
  
class ParamsList(object):
  def __init__(self, xs = None):
    if xs is None:
      self.xs = None
      self.n_updates = 0
    else:
      self.xs = self._copy_params(xs) 
      self.n_updates = 1

  def _copy_param(self, x):
      if isinstance(x, theano.sandbox.cuda.CudaNdarray):
        return to_gpuarray(x, copyif=True)
      elif hasattr(x, 'copy'):
        return x.copy()
      else:
        return x

  def _copy_params(self, xs):
      return [self._copy_param(xi) for xi in xs]

  def iadd(self, ys):
    self.n_updates += 1

    if self.xs is None:
      self.xs = self._copy_params(ys)
    else:
      assert len(self.xs) == len(ys)
      for i, yi in enumerate(ys):
        # these are bizarrely faster
        if hasattr(self.xs[i], 'mul_add'):
           if not isinstance(yi, GPUArray):
             yi = to_gpuarray(yi, copyif=True)
           self.xs[i].mul_add(1.0, yi, 1.0)
        else:
           self.xs[i] += yi
  
  def flatten(self):
    assert self.xs is not None 
    elts_per_subarray = []
    is_array = []
    elt_type = None
    gpu_arrays = [] 
    for w in self.xs:
      if isinstance(w, theano.sandbox.cuda.CudaNdarray):
        w = to_gpuarray(w, copyif=True)
      if isinstance(w, (np.ndarray, GPUArray)):
        if not hasattr(w, 'gpudata'):
          w = pycuda.gpuarray.to_gpu(w)
        elts_per_subarray.append(w.size)
        is_array.append(True)
        if elt_type is None:
          elt_type = w.dtype 
      else:
        assert np.isscalar(w)
        elts_per_subarray.append(1)
        is_array.append(False)
      gpu_arrays.append(w) 
    total_elts = sum(elts_per_subarray)
    result = pycuda.gpuarray.empty((total_elts,), dtype = elt_type)  
    curr_idx = 0
    for (nelts, w) in zip(elts_per_subarray, gpu_arrays):
      if not np.isscalar(w):
        w = w.ravel()
      result_slice = result[curr_idx:(curr_idx + nelts)]
      if isinstance(w, GPUArray):
        pycuda.driver.memcpy_dtod(result_slice.gpudata, w.gpudata, w.nbytes)
      else:
        result_slice.set(w)
      curr_idx += nelts
    assert curr_idx == total_elts  
    return result
  

class Network(object): 
  def __init__(self,  
                     mini_batch_size, 
                     learning_rate, 
                     momentum, 
                     n_filters, 
                     n_out = 10,
                     n_hidden = 300, # originally 500  
                     input_height = 32, 
                     input_width = 32, 
                     n_colors = 3, 
                     conv_activation = 'relu'):
   
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
            poolsize=pool_size, activation = conv_activation)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = ConvPoolLayer(rng, input=layer0.output,
            image_shape=(mini_batch_size, n_filters[0], 14, 14),
            filter_shape=(n_filters[1], n_filters[0], filter_size[0], filter_size[1]), 
            poolsize=pool_size, activation = conv_activation)

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = \
      HiddenLayer(rng, input=layer2_input, n_in=n_filters[1] * filter_size[0] * filter_size[1],
                  n_out=n_hidden, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=n_out)

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

    self.bprop_update = theano.function([x, y], self.grads, updates = updates) 
    self.cost_fn = theano.function([x, y], self.cost)

  def get_weights_list(self):
    return [p.get_value(borrow=True) for p in self.params]

  def get_weights(self):
    return ParamsList(self.get_weights_list()).flatten()
 
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
     if isinstance(w, (GPUArray, np.ndarray)):
       nelts = w.size
       w_flat = w.ravel()
       w_flat += dxs[curr_idx:curr_idx+nelts]
       assert w_flat.strides == dxs.strides
       assert w.ctypes.data == w_flat.ctypes.data 
       p.set_value(w, borrow=True)
     else:
       assert np.isscalar(w)
       nelts = 1 
       p.set_value(w + dxs[curr_idx])
     curr_idx += nelts 

  def set_weights(self, new_w):
    curr_idx = 0
    for p in self.params:
      w = p.get_value(borrow=True)
      if isinstance(new_w, GPUArray):
        nelts = w.size 
        new_reshaped = new_w[curr_idx:curr_idx+nelts].reshape(w.shape)
        if hasattr(w, 'gpudata'):
          pycuda.driver.memcpy_dtod(w.gpudata, new_reshaped.gpudata, new_reshaped.nbytes)
        else:
          p.set_value(new_reshaped.get())
      elif isinstance(new_w, np.ndarray):
        nelts = w.size
        new_reshaped = np.reshape(new_w[curr_idx:curr_idx+nelts], w.shape) 
        p.set_value(new_reshaped)
      else:
        assert np.isscalar(w)
        nelts = 1 
        p.set_value(np.array(new_w[curr_idx:curr_idx+1])[0])
      curr_idx += nelts 
    assert curr_idx == len(new_w)
  
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

  def get_gradients(self, xslice, yslice):
    g_list = [to_gpuarray(g_elt, copyif=True)
              for g_elt in self.bprop_grads(xslice, yslice)]
    return ParamsList(g_list).flatten()

  def average_gradients(self, x, y):
    """
    Get the average gradient across multiple mini-batches
    """
    n_batches = x.shape[0] / self.mini_batch_size
    if n_batches == 1:
      return self.get_gradients(x,y)
    w = 1.0 / n_batches 
    combined = None 
    for batch_idx in xrange(n_batches):
      start = batch_idx*self.mini_batch_size
      stop = start + self.mini_batch_size
      xslice = x[start:stop]
      yslice = y[start:stop]
      g_flat = self.get_gradients(xslice, yslice)
      if combined is None:
        combined = g_flat * w
      else:
        combined = combined.mul_add(1.0, g_flat, w)
    return combined 
  
  def get_state(self, x, y):
    return self.get_weights(), self.average_gradients(x,y)
       
  def update_batches(self, x, y, average=False):
    """
    Returns list containing most recent gradients
    """
    g_avg = ParamsList()
    def fn(xslice, yslice):
      g_list = self.bprop_update(xslice, yslice)
      if average:
        g_avg.iadd(g_list)
        del g_list 
    for_each_slice(x, y, fn, self.mini_batch_size)
    if average:
      return g_avg.flatten() / g_avg.n_updates
    #return np.mean(costs)
    # return np.mean(costs) / self.mini_batch_size 
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
               n_epochs = 20, # how many passes over the data?
               n_out = 10, # how many outputs?  
               n_filters = [64, 25], # how many convolutions in the first two layers of the network?  
               global_learning_rate = 0.1,  # step size for big steps of combined gradients
               local_learning_rate = 0.01,  # step size on each worker
               global_momentum = 0.05,  # momentum of global updates
               local_momentum = 0.05,   # momentum on each worker 
               mini_batch_size = 20,    # how many gradients does a worker average per local step?
               n_local_steps = 2,       # how many mini-batch steps does a worker take?
               approx_local_change = False, # if approx then compare first mini-batch vs. last mini-batch 
               newton_method = 'memoryless-bfgs', # options = 'memoryless-bfgs', 'memoryless-bfgs-avg', 'svd', None
               gradient_average = 'mean', # 'mean', 'best', 'weighted'
               weight_average = 'mean', # 'mean', 'best', 'weighted' 
               global_decay = 0.999, 
               conv_activation = 'relu'): 
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
    self.global_decay = global_decay 
    self.nets = [Network(mini_batch_size = mini_batch_size, 
                    learning_rate = local_learning_rate, 
                    momentum = local_momentum, 
                    n_filters = n_filters, 
                    n_out = n_out, 
                    conv_activation = conv_activation)
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

  def fit(self, train_set_x, train_set_y, shuffle = False, print_frequency = 10000):
    ntrain, ncolors, image_rows, image_cols = train_set_x.shape
    # compute number of minibatches for training, validation and testing
    worker_batch_size = self.mini_batch_size * self.n_local_steps
    simple_backprop = self.n_workers == 1 and self.newton_method is False
    start_time = time.clock()
    def get_shuffled_set():
      start_t = time.time()
      shuffle_indices = np.arange(ntrain)
      np.random.shuffle(shuffle_indices)
      x,y= train_set_x.take(shuffle_indices, axis=0), train_set_y.take(shuffle_indices)
      stop_t = time.time()
      print "Shuffle time", stop_t - start_t 
      return x,y
    # use these to draw validation sets to avoid 
    # overlap with parallel training sets 
    shuffled_x, shuffled_y = get_shuffled_set()   
    
    def get_validation_set():
        validation_start = np.random.randint(0, ntrain - self.mini_batch_size)
        validation_stop = validation_start + self.mini_batch_size
        val_x = shuffled_x[validation_start:validation_stop]
        val_y = shuffled_y[validation_start:validation_stop]
        return val_x, val_y

    for epoch in xrange(self.n_epochs):       
      if shuffle:
        train_set_x, train_set_y = get_shuffled_set()
      start_idx = 0 
      last_print_idx = 0
      last_print_time = time.time()
      # used for momentum 
      dw = None
      while ntrain - start_idx >= worker_batch_size * self.n_workers: 
          
          if not simple_backprop:
            val_x, val_y = get_validation_set()
          ws = []
          gs = []
          ss = []
          ys = []
          costs = []
          for worker_idx  in xrange(self.n_workers):
              # batch_start = start_idx + worker_idx * worker_batch_size
              if worker_idx == 0:
                batch_start = start_idx
              else:
                batch_start = np.random.randint(0, ntrain-worker_batch_size)
              batch_stop = batch_start + worker_batch_size 

              batch_x = train_set_x[batch_start:batch_stop]
              batch_y = train_set_y[batch_start:batch_stop]
              net = self.nets[worker_idx]
              if self.newton_method is not None:
                grad_set_x = batch_x[-self.mini_batch_size:] 
                grad_set_y = batch_y[-self.mini_batch_size:]
                old_w, old_g = net.get_state(grad_set_x, grad_set_y)
              if simple_backprop:
                net.update_batches(batch_x, batch_y, average=False)
              else:
                g_path_avg = net.update_batches(batch_x, batch_y, average=True)
                gs.append(g_path_avg)
              costs.append(net.cost_fn(val_x, val_y))
              if self.newton_method is None:
                ws.append(net.get_weights())
              else:
                last_w, last_g = net.get_state(grad_set_x, grad_set_y)
                ws.append(last_w)
                s = last_w.mul_add(1.0, old_w, -1.0)
                y = last_g.mul_add(1.0, old_g, -1.0)
                ss.append(s)
                ys.append(y)
          if start_idx - last_print_idx >= print_frequency:
            curr_t = time.time()
            #print "  Sample %d / %d, elapsed_time %0.3f" % (start_idx, ntrain, curr_t - last_print_time)
            if len(costs) > 0:
             
              print "  -- worker costs", [float(c) for c in costs]
            last_print_time = curr_t 
            last_print_idx = start_idx 
          # start_idx += worker_batch_size * self.n_workers  # SPLITTING DATA BETWEEN WORKERS SUCKS
          start_idx += worker_batch_size 
          if simple_backprop:
            continue

          if self.gradient_average == 'weighted' or self.weight_average == 'weighted':
            centered_costs = np.array(costs) - np.mean(costs)
            scaled_costs = centered_costs / np.std(costs)
            # a low cost is negative, flip the sign to give it a large weight
            weights = np.exp(-scaled_costs)
            weights /= np.sum(weights) 
           
          lowest_cost_idx = np.argmin(costs)
          if lowest_cost_idx >= len(gs):
            print "Bad lowest cost idx! %d / %d" % (lowest_cost_idx, len(gs))
            lowest_cost_idx = 0
          if self.gradient_average == 'mean':
            g = mean(gs)
          elif self.gradient_average == 'best':
            
            g = gs[lowest_cost_idx]
          elif self.gradient_average == 'weighted':
            g = weighted_mean(gs, weights)
          else:
            assert False, "Not implemented: gradient_average = %s" % self.gradient_average 
          del gs 
          if self.weight_average == 'mean':
            w = mean(ws)
          elif self.weight_average == 'best':
            w = ws[lowest_cost_idx]
          elif self.weight_average == 'weighted':
            w = weighted_mean(ws, weights)
          else:
            assert False, "Not implemented: weight_average = %s" % self.weight_average 
          del ws
 
          # any extra gradient scaling to factor into the learning update 
          rescale_gradient = 1.0 
          if self.newton_method == 'memoryless-bfgs':
              #print "  Starting BFGS"
              #print "  -- gradient type", type(g)
              #print "  -- gradient shape", g.shape
              rhos = []
              alphas = []
              if self.global_learning_rate != 'search':
                norm_before = norm(g) #np.sqrt(dot(g,g))
              for (s,y) in zip(ss,ys):
                  rho = 1.0 / dot(s,y)
                  rhos.append(rho)
                  alpha = rho * dot(s,g) 
                  alphas.append(alpha)
                  g = g.mul_add(1.0, y, -alpha)
              for i in reversed(range(self.n_workers)):
                  s = ss[i]
                  y = ys[i]
                  alpha = alphas[i]
                  rho = rhos[i]
                  beta = rho * dot(y,g)
                  g = g.mul_add(1.0, s, alpha-beta)

              # if we don't have a line search, 
              # should normalize the search direction 
              # since it can grow several orders of magnitude 
              if self.global_learning_rate != 'search':
                norm_after = norm(g) #np.sqrt(dot(g,g))
                rescale_gradient = norm_before / norm_after 
          elif self.newton_method == 'svd':
              
              """
              Original algorithm, before a lot of transposing happened:
              ---------------------------------------------------------
              U, D, V = np.linalg.svd(Y.T, full_matrices=False)
              
              diag_ratios = D / D[0]
              if diag_ratios.min() < cutoff:
                k = argmax(diag_ratios < cutoff)
              else:
                k = len(D)
              U = U[:, :k]
              D = D[:k]
              V = V[:k, :]
              Ug = np.dot(U.T, g)
              DinvUg = np.dot(np.diag(1.0 / D), Ug)
              VDinvUg = np.dot(V.T, DinvUg)
              search_dir = np.dot(S.T, VDinvUg)
              """
              Y = concat(ys) 
              S = concat(ss)
              
              Y = Y.get()
              g = g.get()
               
              V,D,U = np.linalg.svd(Y, full_matrices=False)
              cutoff = 0.0001 * D[0]
              if D.min() < cutoff:
                k = argmax(D < cutoff)
              else:
                k = len(D)
              U = U[:k, :]
              V = V[:, :k]
              D = D[:k]
              
              g = np.dot(U, g)
              g *= (1.0 / D) 
              g = np.dot(V, g)
              g = pycuda.gpuarray.to_gpu(g) 
              g = dot(g, S)
              
          else:
              assert self.newton_method is None, "Unrecognized newton method: %s" % self.newton_method
          if self.global_learning_rate == 'search':
            val_x, val_y = get_validation_set()
            etas = 10.0 ** np.arange(2, -5, -1) 
            ws = []
            w_best = None
            cost_best = np.inf
            # print norm(g) 
            for eta in etas:
              
              w_candidate = w.mul_add(self.global_decay, g, -eta * rescale_gradient)
              
              self.nets[0].set_weights(w_candidate)
              cost = self.nets[0].cost_fn(val_x, val_y)
              # print "   %0.6f ==> %0.6f" % (eta, cost)
              if cost < cost_best:
                cost_best = cost
                w_best = w_candidate
            w = w_best 
          else:
            eta = self.global_learning_rate
            w  = w.mul_add(self.global_decay, g, -eta * rescale_gradient)
          
              
          for worker_idx in xrange(self.n_workers):
            self.nets[worker_idx].set_weights(w)
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
    mean_err = np.mean(errs)
    if np.isnan(mean_err) or np.isinf(mean_err):
      return 0
    else:
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
       n_workers = [2], 
       mini_batch_size = [64], 
       n_local_steps = [ 10 ],  
       global_learning_rate = ['search', 1.0], # global_learning_rates,  # [0.1, 1.0, 2.0], # TODO: 'search'
       local_learning_rate = [0.1], # TODO: 'random'
       global_momentum = [0.0], # TODO: 0.05 
       local_momentum = [0.0], # TODO: 0.05 
       weight_average = ['best',], 
       gradient_average = ['best', 'weighted'],  
       newton_method = [None, 'svd', 'memoryless-bfgs' ], 
       conv_activation = ['relu', 'tanh'],
  ) 

  print "Generated %d parameter combinations" % len(param_combos)
  train_set_x, train_set_y, test_set_x, test_set_y  = \
    load_data(labels='coarse_labels')
  print "Train set:", train_set_x.shape
  print "Test set:", test_set_x.shape
  n_out = len(np.unique(test_set_y))
  n_epochs = 10
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
  
  simple_sgd_ran = True
  for (i, params) in enumerate(param_combos):
    if params['n_workers'] == 1 and params['newton_method'] is None:
      if simple_sgd_ran:
        continue
      else:
        simple_sgd_ran = True 
    param_str = ", ".join("%s = %s" % (k,params[k]) for k in sorted(params))
    print "Param #%d" % (i+1), param_str 
    model = DistLearner(n_epochs = n_epochs, n_out = n_out, **params)

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
  import pickle
  pickle.dump(results, f)
  f.close()
  keys = results.keys()
  values = results.values()
  sorted_indices = sorted(range(len(results)), key = lambda i: results.values()[i])
  for idx in sorted_indices:
    print "%0.3f %s" %   (values[i], keys[i])

