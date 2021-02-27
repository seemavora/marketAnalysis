import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
df = pd.read_csv('../../data/cleanData/oneEncodedDiscord.csv')
X = df[["personal_qualities", "friend_qualities"]]

# ~~~~~ Number of clusters ~~~~~
K = 3
=======
df = pd.read_csv("oneEncodedDiscord.csv")
=======
df = pd.read_csv('../../data/cleanData/oneEncodedDiscord.csv')
>>>>>>> Changes
=======
df = pd.read_csv("oneEncodedDiscord.csv")
>>>>>>> Changes
X = df[["personal_qualities","friend_qualities"]]

# ~~~~~ Number of clusters ~~~~~
K=3
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> Changes
=======
>>>>>>> Changes
=======
>>>>>>> Changes

# ~~~~~ Select random observation as centroids ~~~~~
# Creating a normal scatter plot for the qualities
# using random centroids

# sample() to select some random 5 elements of the data
# to become our 5 initial centroids
Centroids = (X.sample(n=K))

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
plt.scatter(X["personal_qualities"], X["friend_qualities"], c='black')
plt.scatter(Centroids["personal_qualities"],
            Centroids["friend_qualities"], c='red')
=======
plt.scatter(X["personal_qualities"],X["friend_qualities"],c='black')
plt.scatter(Centroids["personal_qualities"],Centroids["friend_qualities"],c='red')
>>>>>>> Changes
=======
plt.scatter(X["personal_qualities"],X["friend_qualities"],c='black')
plt.scatter(Centroids["personal_qualities"],Centroids["friend_qualities"],c='red')
>>>>>>> Changes
=======
plt.scatter(X["personal_qualities"],X["friend_qualities"],c='black')
plt.scatter(Centroids["personal_qualities"],Centroids["friend_qualities"],c='red')
>>>>>>> Changes
plt.title('UNCLUSTERED GRAPH')
plt.xlabel('personal_qualities')
plt.ylabel('friend_qualities')
plt.show()

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<< << << < HEAD
<< << << < HEAD
<< << << < HEAD
# ~~~~~ Create clusters based on the random centroids ~~~~~
== == == =
>>>>>> > Does Kmeans for personal and friends qualities
== == == =
# ~~~~~ Create clusters based on the random centroids ~~~~~
>>>>>> > Elbow method
== == == =
# ~~~~~ Create clusters based on the random centroids ~~~~~
>>>>>> > Changes
=======
<<<<<<< HEAD
# ~~~~~ Create clusters based on the random centroids ~~~~~
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Changes
# ~~~~~ Create clusters based on the random centroids ~~~~~
=======
>>>>>>> Does Kmeans for personal and friends qualities
=======
# ~~~~~ Create clusters based on the random centroids ~~~~~
>>>>>>> Elbow method
<<<<<<< HEAD
>>>>>>> Elbow method
>>>>>>> Changes
diff = 1
j = 0

while(diff != 0):
    XD = X
    i = 1
    for index1, row_c in Centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c["personal_qualities"]-row_d["personal_qualities"])**2
            d2 = (row_c["friend_qualities"]-row_d["friend_qualities"])**2
            d = np.sqrt(d1+d2)
            ED.append(d)
        X[i] = ED
        i = i+1

<<<<<<< HEAD
<< << << < HEAD
<< << << < HEAD
<< << << < HEAD
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
== == == =
>>>>>> > Does Kmeans for personal and friends qualities
== == == =
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
>>>>>> > Elbow method
== == == =
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
>>>>>> > Changes
=======
<<<<<<< HEAD
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
>>>>>>> Changes
    C = []
    for index, row in X.iterrows():
        min_dist = row[1]
        pos = 1
<<<<<<< HEAD
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
=======
>>>>>>> Does Kmeans for personal and friends qualities
=======
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
>>>>>>> Elbow method
=======
# ~~~~~ Create clusters based on the random centroids ~~~~~
=======
>>>>>>> Changes
=======
# ~~~~~ Create clusters based on the random centroids ~~~~~
>>>>>>> Changes
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
>>>>>>> Changes
=======
<<<<<<< HEAD
<<<<<<< HEAD
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
=======
>>>>>>> Does Kmeans for personal and friends qualities
=======
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
>>>>>>> Elbow method
>>>>>>> Changes
=======
# ~~~~~ Recompute centroids for more accurate clustering ~~~~~
>>>>>>> Changes
    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> Elbow method
>>>>>>> Changes
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos = i+1
        C.append(pos)
    X["Cluster"] = C
    Centroids_new = X.groupby(["Cluster"]).mean(
    )[["friend_qualities", "personal_qualities"]]
    if j == 0:
        diff = 1
        j = j+1
    else:
<<<<<<< HEAD
<< << << < HEAD
<< << << < HEAD
<< << << < HEAD
        # The difference between centroids
== == == =
>>>>>> > Does Kmeans for personal and friends qualities
== == == =
        # The difference between centroids
>>>>>> > Elbow method
== == == =
        # The difference between centroids
>>>>>> > Changes
        diff = (Centroids_new['friend_qualities'] - Centroids['friend_qualities']).sum() + (
            Centroids_new['personal_qualities'] - Centroids['personal_qualities']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["friend_qualities","personal_qualities"]]

<<<<<<< HEAD
=======
<<<<<<< HEAD
        # The difference between centroids
        diff = (Centroids_new['friend_qualities'] - Centroids['friend_qualities']).sum() + (
            Centroids_new['personal_qualities'] - Centroids['personal_qualities']).sum()
=======
<<<<<<< HEAD
=======
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
>>>>>>> Changes
<<<<<<< HEAD
        # The difference between centroids
=======
>>>>>>> Does Kmeans for personal and friends qualities
=======
        # The difference between centroids
>>>>>>> Elbow method
        diff = (Centroids_new['friend_qualities'] - Centroids['friend_qualities']).sum() + (Centroids_new['personal_qualities'] - Centroids['personal_qualities']).sum()
<<<<<<< HEAD
>>>>>>> Elbow method
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[
        ["friend_qualities", "personal_qualities"]]

<<<<<<< HEAD
# ~~~~~ Final cluster ~~~~~
# Once diff is 0, we stop training and have the final cluters
color = ['mediumvioletred', 'lightseagreen',
         'lightskyblue', 'mediumslateblue', 'gold']
for k in range(K):
    data = X[X["Cluster"] == k+1]
    plt.scatter(data["personal_qualities"],
                data["friend_qualities"], c=color[k])
plt.scatter(Centroids["personal_qualities"],
            Centroids["friend_qualities"], c='red')
plt.title('CLUSTERED GRAPH')
=======
>>>>>>> Changes
<<<<<<< HEAD
=======
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["friend_qualities","personal_qualities"]]

<<<<<<< HEAD
>>>>>>> Changes
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
# ~~~~~ Final cluster ~~~~~
# Once diff is 0, we stop training and have the final cluters
color=['mediumvioletred','lightseagreen','lightskyblue', 'mediumslateblue', 'gold']
>>>>>>> Changes
=======
>>>>>>> Changes
=======
=======
>>>>>>> Changes
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
        # The difference between centroids
        diff = (Centroids_new['friend_qualities'] - Centroids['friend_qualities']).sum() + (Centroids_new['personal_qualities'] - Centroids['personal_qualities']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["friend_qualities","personal_qualities"]]

# ~~~~~ Final cluster ~~~~~
# Once diff is 0, we stop training and have the final cluters
color=['mediumvioletred','lightseagreen','lightskyblue', 'mediumslateblue', 'gold']
<<<<<<< HEAD
>>>>>>> Changes
=======
>>>>>>> Changes
=======
>>>>>>> Changes
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["personal_qualities"],data["friend_qualities"],c=color[k])
plt.scatter(Centroids["personal_qualities"],Centroids["friend_qualities"],c='red')
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Changes
=======
>>>>>>> Changes
plt.title('CLUSTERED GRAPH')
=======
>>>>>>> Does Kmeans for personal and friends qualities
=======
plt.title('CLUSTERED GRAPH')
>>>>>>> Elbow method
<<<<<<< HEAD
<<<<<<< HEAD
=======
plt.title('CLUSTERED GRAPH')
>>>>>>> Changes
=======
>>>>>>> Elbow method
>>>>>>> Changes
plt.xlabel('personal_qualities')
plt.ylabel('friend_qualities')
plt.show()
=======
plt.title('CLUSTERED GRAPH')
plt.xlabel('personal_qualities')
plt.ylabel('friend_qualities')
plt.show()
>>>>>>> Changes
=======
plt.xlabel('personal_qualities')
plt.ylabel('friend_qualities')
plt.show()
>>>>>>> Changes
=======
plt.title('CLUSTERED GRAPH')
plt.xlabel('personal_qualities')
plt.ylabel('friend_qualities')
plt.show()
>>>>>>> Changes
