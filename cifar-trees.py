import cPickle
import numpy as np
import sklearn  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import SGDClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

with open('conv-train.pickle.0', 'r') as f:
  d = cPickle.load(f)
  x_train = d['fc']
  y_train = np.ravel(d['labels'])

n_samples, n_features = x_train.shape
N = n_samples  
print "Loaded %d training samples w/ %d features" % (n_samples, n_features)
with open('conv-test.pickle.0', 'r') as f:
  d = cPickle.load(f)
  x_test = d['fc']
  y_test = np.ravel(d['labels'])

print "Loaded %d test samples w/ %d features" % x_test.shape



def run(clf, name = None):
  if name is None:
    name = str(clf)
  print "Fitting %s..." % name 
  clf.fit(x_train, y_train)
  print "Evaluating %s..." % name
  y_pred = clf.predict(x_test)
  err = np.mean(y_test != y_pred)
  print 
  print "Error = %0.4f" % (err,)
  print "---"
  print 

#small_rf = RandomForestClassifier(n_jobs = 5, n_estimators = 5, min_samples_leaf = 5, max_depth = 7)
#run(small_rf)


#medium_rf = RandomForestClassifier(n_jobs = 5, n_estimators = 9, min_samples_leaf = 5, max_depth = 11)
#run(medium_rf)

#large_rf = RandomForestClassifier(n_jobs = 5, n_estimators = 13, min_samples_leaf = 5)
#run(large_rf)

class Ensemble(object):
  def __init__(self, n_jobs = 6, n_estimators = 7, n_rf = 10, max_depth = 9):
    self.n_jobs = n_jobs 
    self.n_estimators = n_estimators 
    self.n_rf = n_rf
    self.max_depth = max_depth 
    self.classifiers = []
    self.weights = []

  def __str__(self):
    return "Ensemble(n_estimators = %d, n_rf = %d, max_depth = %d)" % (self.n_estimators, self.n_rf, self.max_depth)
  
  def fit(self, x, y):
    x_wrong = x
    n_wrong = x.shape[0]
    y_wrong = y
    for i in xrange(self.n_rf):
      self.weights.append(x_wrong.shape[0] / float(x.shape[0]))
      clf = RandomForestClassifier(n_jobs = self.n_jobs, n_estimators = self.n_estimators, max_depth = self.max_depth)
      clf.fit(x_wrong,y_wrong)
      self.classifiers.append(clf)
      prob = clf.predict_proba(x_wrong)
      pred = np.argmax(prob, axis = 1)
      maxprob = np.max(prob, axis = 1)
      lowprob = maxprob < 0.5
      wrong = pred != y_wrong
      n_wrong = np.sum(wrong) 
      print "Boosting iteration %d, #wrong: %d, #lowprob: %d, #total:%d" % (i, n_wrong, np.sum(lowprob), x_wrong.shape[0])
      wrong |= lowprob 
      if n_wrong == 0:
        print "Memorized data, aborting!" 
        break 
      x_wrong = x_wrong[wrong]
      y_wrong = y_wrong[wrong]
    
  
  def predict(self, x):
    p = self.classifiers[0].predict_proba(x)
    for c, w in zip(self.classifiers[1:], self.weights[1:]):
      p += w * c.predict_proba(x)
    return np.argmax(p, 1) 

run(Ensemble())
#run(RandomForestClassifier(n_jobs = 6, n_estimators = 201))
#run(ExtraTreesClassifier(n_estimators = 201, n_jobs = 6))
#run(SGDClassifier(loss = "hinge", penalty = "l2", n_iter = int(np.ceil(10**6 / N)), shuffle=True))
#run(AdaBoostClassifier(n_estimators = 201))
#run(GradientBoostingClassifier(n_estimators = 11))

