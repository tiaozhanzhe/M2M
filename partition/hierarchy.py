# encoding: utf-8
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data(path):
	"""
	@brief      Loads a data.
	@param      path  The path
	@return     data set
	"""
	data_set = list()
	with open(path) as f:
		for line in f.readlines():
			if line[0] == '#':
				break
			data = line.strip().split(" ")
			flt_data = list(map(float, data))
			data_set.append(flt_data)
	return data_set

x = load_data('resnet50.txt')
length = len(x)

pca = PCA(n_components=1)
pca.fit(x)

y = np.zeros((length, 2))
y[:, 0] = np.arange(1, length + 1)
y[:, 1] = pca.transform(x)[:, 0]

print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

# plt.scatter(y[:, 0], y[:, 1])
kneighbors_graph = np.zeros((length, length))
for i in range(length - 1):
	kneighbors_graph[i, i+1] = 1
	kneighbors_graph[i+1, i] = 1
	
clustering = AgglomerativeClustering(n_clusters = 4, connectivity = kneighbors_graph).fit(y)

label_pred = clustering.labels_
print(clustering.labels_ )
print(clustering.children_)

color_list = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
# plt.scatter(y[:, 0], y[:, 1])
for i in range(len(y)):
    plt.scatter(y[i][0], y[i][1], c = color_list[label_pred[i]])

plt.show()