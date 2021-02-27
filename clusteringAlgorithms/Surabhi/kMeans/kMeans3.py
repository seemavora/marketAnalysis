import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# data = pd.read_csv('oneEncodedDiscord.csv')
data = pd.read_csv('../../data/rawData/modifiedDiscord.csv')
print(f"Variable:                  Type: \n{data.dtypes}") 

f1 = data['personal_qualities'].values
f2 = data['friend_qualities'].values
X = np.array(list(zip(f1,f2)))

# ~~~~~ Elbow method ~~~~~

def elbowMethod (min, max):
    K_range = range(min,max)
    distortions = []

    for i in K_range:
        kmeanModel = KMeans(n_clusters = i)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    fig1 = plt.figure()
    ex = fig1.add_subplot(111)
    ex.plot(K_range, distortions, 'b*-')

    plt.grid(True)
    plt.ylim([0,15])
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distortion')
    plt.title('Selecting k using Elbow method')
    plt.show()
    

# elbowMethod(1,10)