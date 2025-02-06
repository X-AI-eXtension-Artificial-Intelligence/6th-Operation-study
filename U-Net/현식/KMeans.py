import numpy as np
from sklearn.cluster import KMeans

num_items = 1000
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
print(color_array.shape)

num_classes = 10
label_model = KMeans(n_clusters = num_classes)
label_model.fit(color_array)