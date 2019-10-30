# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
#from time import time
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import metrics
claims = np.loadtxt(open("/Users/Chen/Canopy/cccode/claims_final.csv ","rb"), delimiter=";")
#import pandas as pd 
#path ='/Users/Chen/Canopy/cccode/claims_final.csv'
#claims = pd.read_csv(path, sep=';')
#import csv
#path ='/Users/Chen/Canopy/cccode/claims_final.csv'
#with open(path) as csvDataFile:
#    claims = csv.reader(csvDataFile)
#print(claims)

kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
kmeans.fit(claims)

h = 6
x_min, x_max = claims[:, 0].min() - 1, claims[:, 0].max() + 1
y_min, y_max = claims[:, 1].min() - 1, claims[:, 1].max() + 1
print('claims[:, 0].min()',claims[:, 0].min())
print('claims[:, 0].max()',claims[:, 0].max())
print('claims[:, 1].min()',claims[:, 1].min())
print('claims[:, 1].max()',claims[:, 1].max())
xx,yy= np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h),copy=False)
print(yy.shape)
# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(claims, claims, 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='k', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
