import cPickle
import numpy as np
import sklearn  
from sklearn.cluster import MiniBatchKMeans
# from sklearn.tree import DecisionTreeClassifier 
# from sklearn.linear_model import SGDClassifier 
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import glob 
import sys 
import time 



from PyWiseRF import WiseRF 
Ensemble = WiseRF

train_files = list(glob.glob('/scratch1/imagenet-pickle/train-data.pickle.*'))

KMEANS = False

if len(sys.argv) > 1:
  DEBUG = (sys.argv[1] in ("d", "dbg", "debug"))
else:
  DEBUG = False  

if DEBUG:
  k = 2
  n_trees = 1 
  max_depth = 2
  train_files = train_files[:1]
  max_samples_per_file = 10000
else:
  k = 5
  n_trees = 201
  max_depth = 35
  max_samples_per_file = None
  

coarse_classifiers = []
fine_classifiers = [[] for _ in xrange(k)] 

if KMEANS:
  kmeans = MiniBatchKMeans(n_clusters = k, batch_size = 1000, init_size = 8000, max_iter = 300, n_init = 5, init='random')

for i, filename in enumerate(train_files):
  with open(filename, 'r') as f:
    print "Loading %s..." % filename
    d = cPickle.load(f)
    x_train = d['fc'][:max_samples_per_file, :]
    y_train = np.ravel(d['labels']).astype('int32')[:max_samples_per_file]
   

    print "# samples = %d, # features = %d" % x_train.shape
    n_samples = x_train.shape[0]
    assert n_samples == y_train.shape[0], "x_train: %s, y_train: %s" % (x_train.shape, y_train.shape)
    
    t = time.time()
    if KMEANS:
      if i == 0:
        print "Running clustering..." 
        coarse_y = kmeans.fit_predict(x_train)
      else:
        coarse_y = kmeans.predict(x_train)
      for j in xrange(k):
        print "Num in cluster %d: %d" % (j, np.sum(coarse_y == j))
    else:
      coarse_y = y_train / 100
      coarse_clf = Ensemble(n_estimators = n_trees, max_depth = max_depth)
      print "Training coarse classifier..." 
      coarse_clf.fit(x_train, coarse_y)
      coarse_accuracy = np.mean(coarse_clf.predict(x_train) == coarse_y)
      print "Coarse accuracy = %0.4f" % coarse_accuracy
      coarse_classifiers.append(coarse_clf)
    
    for j in xrange(k):
      mask = (coarse_y == j)
      fine_x = x_train[mask]
      if fine_x.shape[0] > 0:
        print "Training fine classifier for coarse label %d with %d samples" % (j, fine_x.shape[0])
        fine_y = y_train[mask]
        fine_clf = Ensemble(n_estimators = n_trees, max_depth = max_depth)
        fine_clf.fit(fine_x, fine_y)
        # assert fine_clf.n_classes_ == 1000, "Expected 1000 classes but got %d" % fine_clf.n_classes_ 
        fine_classifiers[j].append(fine_clf)
      else:
        print "Skipping coarse label %d" % j 
    print "Elapsed time = %0.4f" % (time.time() - t)
    print 


del x_train 
del y_train 

def predict(x, y = None):
  n_samples = x.shape[0]
  if KMEANS:
    coarse_labels = kmeans.predict(x)
    for i in xrange(k):
      mask = coarse_labels == i
      x_subset = x[mask]
      n_subset = x_subset.shape[0]
      if n_subset > 0:
        for j, fine_clf in enumerate(fine_classifiers[i]):
          fine_pred = fine_clf.predict(x_subset)
          if y is not None:
            print "Accuracy of cluster %d, tree %d: %0.4f" % (i, j, np.mean(fine_pred == y[mask]))
          pred_counts[mask, fine_pred] += 1 
    return np.argmax(pred_counts, axis=1)
  else:  
    coarse_probs = coarse_classifiers[0].predict_proba(x)
    assert coarse_probs.shape == (n_samples, 10)
    for coarse_clf in coarse_classifiers[1:]:
      coarse_probs += coarse_clf.predict_proba(x)
      coarse_probs /= len(coarse_classifiers)
    print "Example coarse probs:", coarse_probs[:20]
    #max_coarse = np.argmax(coarse_probs, axis=1)
    fine_probs = np.empty((n_samples, 1000), dtype='float32')
    prob = np.empty((n_samples, 100), dtype='float32')
    for i in xrange(10):
      prob.fill(0)
      print "Label group %d / 10" % (i+1,)
      curr_clfs = fine_classifiers[i]
      curr_coarse = coarse_probs[:, i]
      for j, fine_clf in enumerate(curr_clfs):
        print "  Classifier %d / %d..." % (j+1, len(curr_clfs)) 
        prob += fine_clf.predict_proba(x)
        #shape of prob is  (n_samples, 100)
        # shape of curr_coarse[:, None] is (n_samples, 1)
      fine_probs[:, start_label:stop_label] = prob * curr_coarse[:, None]
    return np.argmax(fine_probs, axis=1)
    

print "---" 
total_correct = 0 
total_test = 0
for filename in glob.glob('/scratch1/imagenet-pickle/test-data.pickle.*'):
  with open(filename, 'r') as f:
    print "Loading %s..." % filename
    d = cPickle.load(f)
    x_test = d['fc']
    y_test = np.ravel(d['labels']).astype('int32')
    print "# samples = %d, # features = %d" % x_test.shape
    n_samples = x_test.shape[0]
    print "Evaluating..." 
    pred = predict(x_test)
    n_correct = np.sum(y_test == pred)
    print "Accuracy = %0.4f (%d / %d)" % (float(n_correct) / n_samples, n_correct, n_samples)
    total_test += n_samples
    total_correct += n_correct
    print 

print "---" 
print "Overall accuracy = %0.4f (%d / %d)" % (float(total_correct) / total_test, total_correct, total_test)
