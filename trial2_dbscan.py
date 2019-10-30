import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

raw_claims = np.loadtxt(open("/Users/Chen/Canopy/cccode/claims_final.txt","rt"), delimiter=";",dtype ="str")
a,b = raw_claims.shape
for i in range(a):
    if raw_claims[i,4] =='NY':
        raw_claims[i,4] =1
    elif raw_claims[i,4] == 'FL':
        raw_claims[i,4] =2
    elif raw_claims[i,4] == 'CA':
        raw_claims[i,4] =3
    elif raw_claims[i,4] == 'TX':
        raw_claims[i,4] =4
raw_claims = np.array(raw_claims,dtype = 'float')
scale_claims = np.zeros(raw_claims.shape)
for i in range(raw_claims.shape[1]):
    x = raw_claims[:,i]
    scale_claims[:,i]= np.interp(x, (x.min(), x.max()), (0, 1))
'''-----------------------------------------------------'''
claims = np.c_[scale_claims[:,1],scale_claims[:,2]]
print(claims.shape)
#great so far
dbscan = DBSCAN(eps=0.3, min_samples=10)
#n_init = repeat n times with diff initial centroids, find the best. 
clusters = dbscan.fit(claims)
print(clusters.labels_)