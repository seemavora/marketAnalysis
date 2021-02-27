import numpy as np 
import pandas as pd 
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('oneEncodedDiscord.csv', sep=",", header=None)
# print(df.head())

X = df[["personal_qualities","friend_qualities"]]
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

def kmeans(X, k=3, max_iterations=100):
    if isinstance(X, pd.DataFrame):X = X.values
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    P = np.argmin(distance.cdist(X, centroids, 'euclidean'), axis=1)
    for _ in range(max_iterations):
        centroids = np.vstack([X[P==i,:].means(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, 'euclidean'), axis=1)
        if np.array_equal(P,tmp):break
        P = tmp
    return P

P = kmeans(X)

X = sc.inverse_transform(X)
plt.figure(figsize=(15,10))
plt.scatter(X[:,0], X[:,1], c=P)
plt.show()
