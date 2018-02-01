#!/usr/bin/python
'''
#
# knn_random.py
# Visualize the KNN classification result with random data
#
# Author : sosorry
# Date   : 2017/01/12
# Origin : http://docs.opencv.org/2.4/modules/ml/doc/k_nearest_neighbors.html
# Usage  : python knn_random.py
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# A set containing 11 (x,y) values of training data values from 0 to 9
train = np.random.randint(0, 10, (11, 2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
train_labels = np.random.randint(0, 2, (11, 1)).astype(np.float32)

labels = { 0.0:'red', 1.0:'blue' }

# Take Red families and plot them
red = train[train_labels.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 120, 'r', '^')

# Take Blue families and plot them
blue = train[train_labels.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], 120, 'b', 's')

# Add into new comer(test data)
newcomer = np.random.randint(0, 10, (1, 2)).astype(np.float32)
print "newcomer: ", newcomer, "\n"
plt.scatter(newcomer[:, 0], newcomer[:, 1], 300, 'g', 'o')

# Train & find nearest k-Nearest Neighbors
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, results, neighbors, dist = knn.findNearest(newcomer, 3)
# It returns:
# The label given to new-comer depending upon the kNN theory we saw earlier.
#   If you want Nearest Neighbour algorithm, just specify k=1 where k is the number of neighbours.
# The labels of k-Nearest Neighbours.  
# Corresponding distances from new-comer to each nearest neighbour.


print "return: ", ret
print "result: ", results
print "In the ", labels[results[0][0]], "family"
print "neighbors: ", neighbors
i = 0
for arr in neighbors:
    for neighbor in arr:
        print "Neighbor is ", labels[neighbor], " ",
        print "of distance ", dist[0][i]
        i = i+1
print "distance: ", dist

plt.show()
