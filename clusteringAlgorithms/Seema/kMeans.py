import pandas as pd
import numpy as np 
import random as rd
import matplotlib.pyplot as plt

data = pd.read_csv('../../data/cleanData/oneEncodedDiscord.csv')
X = data[['friend_qualities', 'personal_qualities']]
# plt.scatter(X['major_department'], X['school_balance'], c= 'black')
# plt.xlabel('Major')
# plt.ylabel('School Balance')
# plt.show()

#number of clusters 
K = 3

Centroids = (X.sample(n=K))
plt.scatter(X['friend_qualities'], X['personal_qualities'], c= 'black')
plt.scatter(Centroids['friend_qualities'], Centroids['personal_qualities'], c= 'red')
plt.xlabel('Major')
plt.ylabel('School Balance')
# plt.show()

# Step 3 - Assign all the points to the closest cluster centroid
# Step 4 - Recompute centroids of newly formed clusters
# Step 5 - Repeat step 3 and 4

diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["friend_qualities"]-row_d["friend_qualities"])**2
            d2=(row_c["personal_qualities"]-row_d["personal_qualities"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

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
    Centroids_new = X.groupby(["Cluster"]).mean()[["personal_qualities","friend_qualities"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['personal_qualities'] - Centroids['personal_qualities']).sum() + (Centroids_new['friend_qualities'] - Centroids['friend_qualities']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["personal_qualities","friend_qualities"]]

color=['blue','green','cyan']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["friend_qualities"],data["personal_qualities"],c=color[k])
plt.scatter(Centroids["friend_qualities"],Centroids["personal_qualities"],c='red')
plt.xlabel('Friend Qualities')
plt.ylabel('Personal Qualities')
plt.show()