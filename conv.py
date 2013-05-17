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

class Accumulator(object):
  def __init__(self):
    self.arrays = []
    self.n_updates = 0

  def add(self, xs):
    self.n_updates += 1
    if len(self.arrays) == 0:
      self.arrays = xs
    else:
      self.arrays = [newx + oldx for (newx, oldx) in zip(xs, self.arrays)]

  def sums(self):
    return self.arrays
  
  def averages(self):
    return [x / self.n_updates for x in self.arrays]

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
    print '... building the model'

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
    
    self.validate_model = theano.function([x,y], layer3.errors(y)) 
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
    self.bprop = theano.function([x, y], self.grads)



  def get_weight_arrays(self):
    return [p.get_value(borrow=True) for p in self.params]

  def update_step(self, grads, old_dxs):
    new_dxs = []
    for (i, p) in enumerate(self.params):
      g = grads[i]
      old_dx = old_dxs[i]
      dx = -np.array(g) * self.learning_rate
      dx += self.momentum * old_dx 
      new_dxs.append(dx)   
      p.set_value(p.get_value(borrow = True) + dx)
    return new_dxs 
 
  def for_each_slice(self, x, y, fn):
    for mini_batch_idx in xrange(x.shape[0] / self.mini_batch_size):
      start = mini_batch_idx * self.mini_batch_size 
      stop = start + self.mini_batch_size 
      xslice = x[start:stop]
      yslice = y[start:stop]
      fn(xslice, yslice)

  def average_gradients(self, x, y):
    """
    Get the average gradient across multiple mini-batches
    """
    combined_grads = []
    acc = Accumulator()
    def fn(xslice, yslice):
      acc.add([np.array(g) for g in self.bprop(xslice, yslice)])
    self.for_each_slice(x, y, fn)
    return acc.averages()
  
 
  def update_batches(self, x, y):
    initial_weights = [p.get_value(borrow=False) for p in self.params]
    initial_grads = self.average_gradients(x,y) 
    dxs = [0.0 for _ in initial_grads]
    def fn(xslice, yslice):
      grads = self.bprop(xslice, yslice)
      new_dxs = self.update_step(grads, dxs)
      del dxs[0:len(dxs)]
      dxs.extend(new_dxs)
    self.for_each_slice(x, y, fn)
    final_weights = self.get_weight_arrays()
    final_gradients = self.average_gradients(x,y)
    change_in_weights = [new_w - old_w for (new_w, old_w) in 
                         zip(initial_weights, final_weights)]
    change_in_gradients = [new_g - old_g for (new_g, old_g) in 
                           zip(initial_grads, final_gradients)]
    #print self.cost
    return final_weights, final_gradients, change_in_weights, change_in_gradients 

def dot(array_list1, array_list2):
  dots = [(x*y).sum() for x,y in zip(array_list1, array_list2)]
  return np.sum(dots)

def norm(array_list):
  return np.sqrt(dot(array_list, array_list))


def evaluate(global_learning_rate = None, # if None, then do a line search  
             local_learning_rate=0.01, 
             global_momentum = 0, 
             local_momentum = 0.01, 
             n_workers = 2,         
             newton = 'one-shot-lbfgs', 
             worker_batch_size = 100, 
             mini_batch_size = 100, 
             n_epochs=200,
             n_filters=[20, 50],
             line_search_prct = 0.05,  
             validation_prct = 0.05):


    xtrain, ytrain, test_set_x, test_set_y  = load_data(labels='coarse_labels')
    n_out = len(np.unique(test_set_y))
 
    ntrain, ncolors, image_rows, image_cols = xtrain.shape
    # split up the training set into actual training data 
    # and validation 
    train_indices = np.arange(ntrain)
    np.random.shuffle(train_indices)
    nvalid = int(ntrain * validation_prct)
  
    train_set_x = xtrain[train_indices[:-nvalid], :, :, :]
    train_set_y = ytrain[train_indices[:-nvalid]]
    valid_set_x = xtrain[train_indices[-nvalid:], :, :, :]
    valid_set_y = ytrain[train_indices[-nvalid:]]
    ntest = ytrain.shape[0]
    print "Train set:", train_set_x.shape
    print "Validation set:", valid_set_x.shape
    print "Test set:", test_set_x.shape
   
    #train_set_x = theano.shared(train_set_x)
    #train_set_y = theano.shared(train_set_y)
    #test_set_x = theano.shared(test_set_x)
    #test_set_y = theano.shared(test_set_y) 
   
    # compute number of minibatches for training, validation and testing
    n_test_batches = ntest / mini_batch_size 
    n_train_batches = (ntrain-nvalid) / mini_batch_size
    n_valid_batches = nvalid / mini_batch_size

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    net = Network(mini_batch_size = mini_batch_size, 
                  learning_rate = local_learning_rate, 
                  momentum = local_momentum, 
                  n_filters = n_filters, 
                  n_out = n_out)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        shuffle_indices = np.arange(len(train_set_x))
        np.random.shuffle(shuffle_indices)
        train_set_x = train_set_x[shuffle_indices]
        train_set_y = train_set_y[shuffle_indices]
        for i in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + i

           
            minibatch_x = train_set_x[i*mini_batch_size:(i+1)*mini_batch_size, :, :, :]
            minibatch_y = train_set_y[i*mini_batch_size:(i+1)*mini_batch_size]         
            # cost_ij = net.train_model(minibatch_x, minibatch_y)
            w,g,s,y = net.update_batches(minibatch_x, minibatch_y)
            if iter % 100 == 0: 
              print "Change in weights: ", norm(s)
              print "Change in gradients: ", norm(y)
              #test_score = run_model(net.test_model, test_set_x, test_set_y)
              #print "     epoch %i, minibatch %i/%i, test error of best model %f %%" %\
              #            (epoch, i + 1, n_train_batches, test_score * 100.)
               

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return best_validation_loss * 100.

if __name__ == '__main__':
    evaluate(local_learning_rate = 0.1, local_momentum = 0.05)
    for n_workers in [2]:
      for local_batch_size in [100, 200]:
        for mini_batch_size in [10, 50]:
          for local_learning_rate in ['random', 0.01, 0.001]:
            for global_learning_rate in ['search', 0.01, 0.001]: 
              for combine_weights in ['weighted', 'mean']:
                for combine_gradients in ['weighted', 'mean']:
                  for newton in ['one-shot-lbfgs', 'global-lbfgs-1', 'global-lbfgs-5', None]:  
                    pass 


