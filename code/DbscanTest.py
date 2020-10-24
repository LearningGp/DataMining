from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import dbscan
import numpy as np

X, _ = datasets.make_moons(500, noise=0.1, random_state=1)
df = pd.DataFrame(X, columns=['x', 'y'])
df.plot.scatter('x', 'y', s = 200, alpha = 0.5, c = "green" , title = 'dataset by DBSCAN')
plt.show()

core_samples, cluster_ids = dbscan(X, eps=0.2, min_samples=20)

df = pd.DataFrame(np.c_[X, cluster_ids], columns=['x', 'y', 'cluster_id'])
df['cluster_id'] = df['cluster_id'].astype('i2')

df.plot.scatter('x', 'y', s=200, c=list(df['cluster_id']), cmap="Reds", colorbar=False, alpha=0.6, title='DBSCAN cluster result')
plt.show()