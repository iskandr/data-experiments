import numpy as np 

def permute_dim(x, d):
    vec = x[:, d]
    unique_vals = np.unique(vec)
    perm = np.random.permutation(unique_vals)
    xp = x.copy()
    for i, val in enumerate(unique_vals):
        xp[vec == val, d] = perm[i]
    return xp 
    
def split_data(x,y):
    r = np.random.randn(x.shape[0])
    mask = r > 0
    xtrain = x[mask, :]
    ytrain = y[mask]
    xtest = x[~mask, :]
    ytest = y[~mask]
    return xtrain, xtest, ytrain, ytest, mask
    

def label_error(ypred, ytest):
    return np.sum(ypred != ytest) / float(len(ytest))
    
def mean_sum_squared_error(ypred, ytest):
    return np.sum( (ypred - ytest) ** 2) / len(ytest)

def univariate_stats(x,y):
    nrows, ndims = x.shape
    errors = np.zeros(ndims)
    is_float = (y.dtype == 'float')

    for d in xrange(ndims):
        vec = x[:, d]
        best_err = np.inf
        if is_float:
            for t in np.unique(vec)[:-1]:
                mask = (vec <= t)
                err1 = label_error(mask, y)
                err2 = label_error(mask, y)
                if err1 < best_err: best_err = err1
                if err2 < best_err: best_err = err2
        else:
            pred =        
        errors[d] = best_err
    return errors
    
    
  
