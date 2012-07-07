import numpy as np 
import sklearn 


def generate_data(n_features = 500, n_classes = 10, n_rows_per_class = 5000): 
  Xs = []
  Ys = []
  for i in xrange(n_classes):
    M = np.random.randn(n_features) * 0.5
    S = np.diag(np.random.uniform(0.1, 5, n_features))
    Xs.append(np.random.multivariate_normal(M, S, n_rows_per_class))
    Ys.append(np.ones(n_rows_per_class, dtype='int') * i)
  
  X = np.vstack(Xs)
  Y = np.hstack(Ys)
  random_indices = np.random.permutation(np.arange(len(Y)))
  X = X[random_indices, :]
  Y = Y[random_indices] 
  return X,Y
  

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, \
   RandomForestClassifier, ExtraTreesClassifier
from sklearn.qda import QDA
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.svm import LinearSVC 
from regularized import RegularizedQDA, RegularizedGaussianNB

def mk_classifiers(n_samples):
  log_samples = int(np.ceil(np.log(n_samples)))
  sqrt_samples = int(np.sqrt(n_samples))

      
  #everything except the SGD learners, which we'll add next 
  classifiers = {
      'Gaussian Naive Bayes' : GaussianNB(),
      'Regularized Gaussian Naive Bayes' : RegularizedGaussianNB(extra_variance=0.1),
      'Decision Tree (min_samples_leaf = 1)' : DecisionTreeClassifier(),  
      'Decision Tree (min_samples_leaf = %d)'  % log_samples : 
        DecisionTreeClassifier(min_samples_split = log_samples), 
      'Decision Tree (min_samples_leaf = %d)'  % sqrt_samples : 
        DecisionTreeClassifier(min_samples_split = sqrt_samples), 
    
      'LDA' : LDA(), 
      'QDA' : QDA(), 
      'Regularized QDA': RegularizedQDA(avg_weight = 0.1),
      
      'Logistic Regression (liblinear)' : LogisticRegression(), 
      'Linear SVM (liblinear)' : LinearSVC(), 
  }
  def mk_rf(criterion, min_samples_leaf,  n_trees):
    return RandomForestClassifier(n_estimators= n_trees, \
      compute_importances=False, \
      min_samples_leaf = min_samples_leaf, \
      criterion = criterion)
  def mk_boost(subsample, max_depth, n_estimators):
    return GradientBoostingClassifier(max_depth = max_depth, \
      n_estimators = n_estimators, 
      subsample = subsample) 
  for n_trees in [2,4,8,16,32,64,128]:
      classifiers['Random Forest (min_samples_leaf = 1, %d trees)' % n_trees] = mk_rf('gini', 1, n_trees)
      classifiers['Random Forest (min_samples_leaf = %d, %d trees)' % (log_samples, n_trees)] = mk_rf('gini', log_samples, n_trees)
      classifiers['Random Forest (min_samples_leaf = %d, %d trees)' % (sqrt_samples, n_trees)] = mk_rf('gini', sqrt_samples, n_trees)
      classifiers['Extra-Trees (%d trees)' % n_trees] = ExtraTreesClassifier(n_estimators = n_trees)
      classifiers['Gradient Boosting (sample size = 100%%, %d stumps)' % n_trees] = mk_boost(1.0, 3, n_trees) 
      classifiers['Gradient Boosting (sample size = 25%%, %d stumps)' % n_trees] = mk_boost(0.25, 3, n_trees)

  
  for u in [2 * 10**6, 10**6, 250000]:
    n_iter = int(np.ceil(u / float(n_samples)))
    classifiers['Linear SVM (SGD, %d updates / %d epochs)' % (u, n_iter)] = SGDClassifier(loss="hinge", n_iter = n_iter, shuffle=True)
    classifiers['Logisic Regression (SGD, %d updates / %d epochs)' % (u, n_iter)] = SGDClassifier(loss="log", n_iter = n_iter, shuffle = True)    
    if n_iter == 1: break
  return classifiers 

def accuracy(Ypred, Y): 
  return np.sum(Ypred == Y) / float(len(Y))

def random_split(X,Y):
  mask = np.random.uniform(0,1, X.shape[0]) > 0.5
  Xtrain = X[mask, :]
  Xtest = X[~mask, :]
  Ytrain = Y[mask]
  Ytest = Y[~mask]
  return Xtrain, Ytrain, Xtest, Ytest

from collections import namedtuple 
import timeit 
import copy 
def get_timings(X,Y, Xtest = None, Ytest = None): 
  n, d = X.shape
  
  if Xtest is None:
    Xtrain, Ytrain, Xtest, Ytest = random_split(X, Y)
  else:
    assert Ytest is not None
    Xtrain = X
    Ytrain = Y

  n_train = Xtrain.shape[0]
  n_test = Xtest.shape[0]
  
  assert d == Xtest.shape[1]
  assert n_train == len(Ytrain)
  assert n_test == len(Ytest)
  
  print "n_train = %d, n_test = %d, n_dims = %d" % (n_train, n_test, d)
  unique_labels = np.unique(Ytrain)
  max_count = 0
  max_label = 0
  for l in unique_labels:
    c = np.sum(Ytrain == l)
    if c > max_count: 
      max_count = c 
      max_label = l
  prob_in_training = max_count / float(len(Ytrain))
  prob_in_test = np.sum(Ytest == max_label) / float(len(Ytest))
  print "n_classes = %d, most likely class = %d, p(%d in training) = %s, p(%d in test) = %s" % \
    (len(np.unique(Ytrain)), max_label, max_label, prob_in_training, max_label, prob_in_test) 
  print 
  learning_algorithms = mk_classifiers(n_train) 
  results = {} 
  Result = namedtuple('Result', ('train_time', 'test_time', 'total_time', 'train_accuracy', 'test_accuracy', 'precision', 'recall'))
  for name, model in sorted(learning_algorithms.items()):
    # copy the model so it doesn't take up space after we use it 
    model = copy.deepcopy(model)
    print name
    result = None
    try:
      t0 = timeit.default_timer()
      model.fit(Xtrain, Ytrain)
      t1 = timeit.default_timer()
      test_pred = model.predict(Xtest)
      t2 = timeit.default_timer()
      train_pred = model.predict(Xtrain)
      
      train_accuracy = accuracy(train_pred, Ytrain)
      test_accuracy = accuracy(test_pred, Ytest)
      
      pred_nnz = float(np.sum(test_pred != 0)) 
      y_nnz = float(np.sum(Ytest != 0)) 
      correct_nz = (test_pred != 0) & (test_pred == Ytest)
      correct_nnz = np.sum(correct_nz)
      precision = correct_nnz / pred_nnz
      recall = correct_nnz / y_nnz
      train_time = t1 - t0
      test_time = t2 - t1
      result = Result(train_time, test_time, train_time + test_time, train_accuracy, test_accuracy, precision, recall)
    except (np.linalg.LinAlgError, AttributeError, MemoryError, OverflowError, ZeroDivisionError, AssertionError):
       print "Failed: ", sys.exc_info()
    results[name] = result 
    print result 
    print 
  return results
  
import sys 
def run(Xtrain = None, Ytrain = None, Xtest = None, Ytest = None):
  if Xtrain is None:
    assert Ytrain is None
    assert Xtest is None
    assert Ytest is None
    print "Generating data..."
    Xtrain, Ytrain = generate_data()
  results = get_timings(Xtrain, Ytrain, Xtest, Ytest)
  print "<table border='2'>"
  print "<tbody>"
  print "<tr>"
  print "  <th>Algorithm</th>"
  print "  <th>Training / Test Time</th>"
  print "  <th>Training / Test Accuracy</th>"
  print "</tr>"
  
  for name, res in sorted(results.items()):
    if res is None:
      print "<tr><td class='name'>%s</td><td colspan='2'><em>failed</em></td></tr>" % name
    else:
      print "<tr><td class='name'>%s</td><td>%.2fs / %.2fs</td><td>%.2f%% / %.2f%%</td></tr>" % \
        (name, res.train_time, res.test_time, res.train_accuracy * 100, res.test_accuracy * 100)
  print "</tbody>"
  print "</table>"
  return results 
