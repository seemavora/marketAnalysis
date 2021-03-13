import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('../../data/cleanData/oneEncodedDiscord.csv')
X = df[["personal_qualities","friend_qualities"]]

# ~~~~~ Number of clusters ~~~~~
K=3

# ~~~~~ Select random observation as centroids ~~~~~
# Creating a normal scatter plot for the qualities
# using random centroids

# sample() to select some random 5 elements of the data
# to become our 5 initial centroids
Centroids = (X.sample(n=K))

plt.scatter(X["personal_qualities"],X["friend_qualities"],c='black')
plt.scatter(Centroids["personal_qualities"],Centroids["friend_qualities"],c='red')
plt.title('UNCLUSTERED GRAPH')
plt.xlabel('personal_qualities')
plt.ylabel('friend_qualities')
plt.show()

<<<<<<< HEAD
<<<<<<< HEAD
# ~~~~~ Create clusters based on the random centroids ~~~~~
=======
>>>>>>> Does Kmeans for personal and friends qualities
=======
# ~~~~~ Create clusters based on the random centroids ~~~~~
>>>>>>> Elbow method
diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["personal_qualities"]-row_d["personal_qualities"])**2
            d2=(row_c["friend_qualities"]-row_d["friend_qualities"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

<<<<<<< HEAD
<<<<<<< HEAD
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
=======
>>>>>>> Does Kmeans for personal and friends qualities
=======
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
>>>>>>> Elbow method
    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["friend_qualities","personal_qualities"]]
    if j == 0:
        diff=1
        j=j+1
    else:
<<<<<<< HEAD
<<<<<<< HEAD
        # The difference between centroids
=======
>>>>>>> Does Kmeans for personal and friends qualities
=======
        # The difference between centroids
>>>>>>> Elbow method
        diff = (Centroids_new['friend_qualities'] - Centroids['friend_qualities']).sum() + (Centroids_new['personal_qualities'] - Centroids['personal_qualities']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["friend_qualities","personal_qualities"]]

<<<<<<< HEAD
<<<<<<< HEAD
# ~~~~~ Final cluster ~~~~~
# Once diff is 0, we stop training and have the final cluters
color=['mediumvioletred','lightseagreen','lightskyblue', 'mediumslateblue', 'gold']
=======
color=['blue','green','cyan', 'purple', 'black']
>>>>>>> Does Kmeans for personal and friends qualities
=======
# ~~~~~ Final cluster ~~~~~
# Once diff is 0, we stop training and have the final cluters
color=['mediumvioletred','lightseagreen','lightskyblue', 'mediumslateblue', 'gold']
>>>>>>> Elbow method
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["personal_qualities"],data["friend_qualities"],c=color[k])
plt.scatter(Centroids["personal_qualities"],Centroids["friend_qualities"],c='red')
<<<<<<< HEAD
<<<<<<< HEAD
plt.title('CLUSTERED GRAPH')
=======
>>>>>>> Does Kmeans for personal and friends qualities
=======
plt.title('CLUSTERED GRAPH')
>>>>>>> Elbow method
plt.xlabel('personal_qualities')
plt.ylabel('friend_qualities')
plt.show()