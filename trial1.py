#import csv
import numpy as np
from sklearn.cluster import KMeans
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
''' -------------------flexible below ------------------------------'''
#0:2 means 0 and 1, not including the right boud
#X = np.reshape(X, (-1, 1))
claims = np.c_[scale_claims[:,6],scale_claims[:,7]]
print(claims.shape)
#great so far
kmeans = KMeans(init='k-means++', n_clusters=25, n_init=3)
#n_init = repeat n times with diff initial centroids, find the best. 
kmeans.fit(claims)
#so far makes sense.
x_step = 0.001#Step of x and y axis
y_step = 0.001
x_min, x_max = claims[:, 0].min() - x_step, claims[:, 0].max() + x_step
y_min, y_max = claims[:, 1].min() - y_step, claims[:, 1].max() + y_step
#here only accounted for the first two column: family ID and family member ID
print('claims[:, 0].min()',claims[:, 0].min())
print('claims[:, 0].max()',claims[:, 0].max())
print('claims[:, 1].min()',claims[:, 1].min())
print('claims[:, 1].max()',claims[:, 1].max())
#output: family member ID goes to 61... that's weired.. isn't it? ok look at this feature first.
xx,yy= np.meshgrid(np.arange(x_min, x_max, x_step), np.arange(y_min, y_max, y_step),copy=False)
print(yy.shape)
print(xx.shape)
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
#dk what it "Paired"
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(claims[:,0], claims[:,1], 'k.', markersize=5)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on Family Member ID versus Provider ID\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
