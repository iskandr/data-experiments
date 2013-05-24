import time 
import numpy as np 

import pycuda
import pycuda.autoinit 

from striate import ConvNet, dot, norm, concat  

class DistConvNet(object):
  def __init__(self, 
               n_workers = 1,
               n_epochs = 10, # how many passes over the data?
               pretrain_epochs = 0, # how many pure SGD passes?
               posttrain_epochs = 10,  # how many cleanup SGD passes?  
               n_out = 10, # how many outputs?  
               n_filters = [64, 96], # how many convolutions in the first two layers of the network?  
               global_learning_rate = 'search',  # step size for big steps of combined gradients
               local_learning_rate = 0.1,  # step size on each worker
               global_momentum = 0.05,  # momentum of global updates
               local_momentum = 0.05,   # momentum on each worker 
               mini_batch_size = 20,    # how many gradients does a worker average per local step?
               n_local_steps = 2,       # how many mini-batch steps does a worker take?
               approx_local_change = False, # if approx then compare first mini-batch vs. last mini-batch 
               newton_method = 'memoryless-bfgs', # options = 'memoryless-bfgs', 'memoryless-bfgs-avg', 'svd', None
               gradient_average = 'mean', # 'mean', 'best', 'weighted'
               weight_average = 'mean', # 'mean', 'best', 'weighted' 
               global_decay = 0.9995, 
               conv_activation = 'relu'): 
    self.n_workers = n_workers
    self.n_epochs = n_epochs
    self.pretrain_epochs = pretrain_epochs
    self.posttrain_epochs = posttrain_epochs 
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
    self.nets = [ConvNet(batch_size = mini_batch_size, 
                    learning_rate = local_learning_rate, 
                    momentum = local_momentum, 
                    n_filters = n_filters, 
                    input_size = (32,32),
                    n_out = n_out, 
                    conv_activation = conv_activation)
                 for _ in xrange(self.n_workers)]
    

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
    start_time = time.clock()
    def get_shuffled_set():
      shuffle_indices = np.arange(ntrain)
      np.random.shuffle(shuffle_indices)
      return train_set_x.take(shuffle_indices, axis=0), train_set_y.take(shuffle_indices)
    # use these to draw validation sets to avoid 
    # overlap with parallel training sets 
    shuffled_x, shuffled_y = get_shuffled_set()  
 
    n_acc_val = 2000 
    acc_val_x, acc_val_y = shuffled_x[:n_acc_val], shuffled_y[:n_acc_val]
    shuffled_x, shuffled_y = shuffled_x[n_acc_val:], shuffled_y[n_acc_val:]
    
    def get_validation_set(size = 200):
        validation_start = np.random.randint(0, len(shuffled_y) - size)
        validation_stop = validation_start + size 
        val_x = shuffled_x[validation_start:validation_stop]
        val_y = shuffled_y[validation_start:validation_stop]
        return val_x, val_y

    # first run some pure SGD epochs on a single worker 
    for epoch in xrange(self.pretrain_epochs):
      print "Pretraining epoch", epoch 
      if shuffle: train_set_x, train_set_y = get_shuffled_set()
      self.nets[0].fit(train_set_x, train_set_y)
      print "  -- validation accuracy = %0.3f" % (self.score(acc_val_x, acc_val_y) * 100)
    
    if self.pretrain_epochs > 0 and self.n_workers > 1:
      w = self.nets[0].get_weights()
      print "Weight vector len = %d" % len(w) 
      for i in xrange(1, self.n_workers):
        self.nets[0].set_weights(w)
     
    for epoch in xrange(self.n_epochs):       
      if shuffle: train_set_x, train_set_y = get_shuffled_set()
      start_idx = 0 
      # will eventually be used for momentum 
      dw = None
      while ntrain - start_idx >= worker_batch_size: 
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
              g_path_avg = net.fit(batch_x, batch_y, return_average_gradient=True)
              gs.append(g_path_avg)
              costs.append(net.average_cost(val_x, val_y))
              if self.newton_method is None:
                ws.append(net.get_weights())
              else:
                last_w, last_g = net.get_state(grad_set_x, grad_set_y)
                ws.append(last_w)
                s = last_w.mul_add(1.0, old_w, -1.0)
                y = last_g.mul_add(1.0, old_g, -1.0)
                ss.append(s)
                ys.append(y)
          
          start_idx += worker_batch_size 
 
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
          else:
            assert self.gradient_average == 'weighted'
            g = weighted_mean(gs, weights)
          del gs
  
          if self.weight_average == 'mean':
            w = mean(ws)
          elif self.weight_average == 'best':
            w = ws[lowest_cost_idx]
          else:
            assert self.weight_average == 'weighted'
            w = weighted_mean(ws, weights)
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
                norm_after = norm(g) 
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
            etas = [320, 160, 80, 40, 20, 10, 5, 1, 0.5, .1, .05, .01, ]
            ws = []
            w_best = None
            eta_best = None
            cost_best = np.inf
            for eta in etas:
              w_candidate = w.mul_add(self.global_decay, g, -eta * rescale_gradient)
              
              self.nets[0].set_weights(w_candidate)
              cost = self.nets[0].average_cost(val_x, val_y)
              # print "   %0.6f ==> %0.6f" % (eta, cost)
              if cost < cost_best:
                cost_best = cost
                w_best = w_candidate
                eta_best = eta 
            w = w_best
            print "  -- best step size", eta_best, "step norm", (eta*rescale_gradient)*norm(g)
          else:
            eta = self.global_learning_rate
            w  = w.mul_add(self.global_decay, g, -eta * rescale_gradient)
          for worker_idx in xrange(self.n_workers):
            self.nets[worker_idx].set_weights(w)
      print "Epoch %d -- validation accuracy = %0.3f" % (epoch, self.score(acc_val_x, acc_val_y) * 100)

    for epoch in xrange(self.posttrain_epochs):
      print "Posttraining epoch", epoch 
      train_set_x, train_set_y = get_shuffled_set()
      self.nets[0].fit(train_set_x, train_set_y)
      print "  -- validation accuracy = %0.3f" % (self.score(acc_val_x, acc_val_y) * 100)
    end_time = time.clock()
    elapsed = end_time - start_time 
    return elapsed 

  def score(self, test_set_x, test_set_y):
    """
    Return average accuracy on the test set
    """
    mean_err = self.nets[0].average_error(test_set_x, test_set_y)
    if np.isnan(mean_err) or np.isinf(mean_err):
      return 0
    else:
      return 1 - mean_err
