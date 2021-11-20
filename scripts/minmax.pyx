import numpy as np
from scipy.spatial.distance import pdist, squareform

def min_max_product(A,B, transposed_B=False):
     if not transposed_B:
          B = np.transpose(B)
     Y = np.zeros((len(B),len(A)))
     cdef int i = 0
     for i in range(len(B)):
          Y[i] = np.maximum(A, B[i]).min(1)
     return np.transpose(Y)

def get_matrix(labels, feats):
     classes = np.unique(labels)

     dm = squareform(pdist(feats))
     dm2 = dm.copy()
     cdef int cl = 0
     for cl in classes:
          dom = dm[np.ix_(labels==cl,labels!=cl)]
          dm[np.ix_(labels==cl,labels==cl)] = \
               min_max_product(dom, dom, transposed_B=True)
     dm = np.maximum(dm, dm2)
     return dm