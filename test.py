from sklearn.cluster import DBSCAN
import numpy as np
X = np.array([[1, 2,3], [2, 2,2], [2, 3,2],[8, 7,2], [8, 8,3], [25, 80,4]])
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
print(clustering.labels_)

