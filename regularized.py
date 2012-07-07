import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.qda import QDA

class RegularizedQDA:
  """
    Three types of regularization are possible:
    - regularized the covariance of a class toward the 
      average variance within that class
    - regularize the covariance of a class toward the
      pooled covariance across all classes
    - add some constant amount of variance to each feature
  """
  def __init__(self, avg_weight = 0.1, pooled_weight = 0, extra_variance = 0):
    self.avg_weight = avg_weight
    self.pooled_weight = pooled_weight
    self.extra_variance = extra_variance 
    self.model = QDA()
    
  def fit(self, X, Y):
    self.model.fit(X,Y)
    I = np.eye(X.shape[1])
    a = self.avg_weight
    p = self.pooled_weight
    ev = self.extra_variance 
    original_weight = 1.0 - a - p
    scaled_pooled_cov = p * np.cov(X.T)
    assert scaled_pooled_cov.shape == I.shape
    assert all([C.shape == I.shape for C in self.model.rotations])
    self.model.rotations = \
      [original_weight * C + \
       a * np.mean(np.diag(C)) * I + \
       scaled_pooled_cov + ev * I \
       for C in self.model.rotations] 
      
  def predict(self, X):
    return self.model.predict(X)
    

class RegularizedGaussianNB:
  """
  Three types of regularization are possible:
    - regularized the variance of a feature within a class toward the 
      average variance of all features from that class
    - regularize the variance of a feature within a class toward its
      pooled variance across all classes
    - add some constant amount of variance to each feature
  In practice, the latter seems to work the best, though the regularization
  value should be cross-validated. 
  """
  def __init__(self, avg_weight = 0, pooled_weight = 0, extra_variance = 0.1):
    self.pooled_weight = pooled_weight
    self.avg_weight = avg_weight
    self.extra_variance = extra_variance
    self.model = GaussianNB()
    
  def fit(self, X,Y):
    self.model.fit(X,Y)
    p = self.pooled_weight
    a = self.avg_weight
    ev = self.extra_variance 
    original_weight = 1.0 - p - a
    pooled_variances = np.var(X, 0)
    for i in xrange(self.model.sigma_.shape[0]):
      class_variances = self.model.sigma_[i, :]
      new_variances = original_weight*class_variances + \
        p * pooled_variances + \
        a * np.mean(class_variances) + \
        ev 
      self.model.sigma_[i, :] = new_variances
        
        
  def predict(self, X):
    return self.model.predict(X)
      
    
