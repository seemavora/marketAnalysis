import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("oneEncodedDiscord.csv")
# print(df.head())

# plt.figure(figsize=(10,6))
# plt.title("Survey Results")
# sns.axes_style("dark")
# sns.violinplot(y=df["music_genre"])
# plt.show()

# plt.figure(figsize=(15,6))
# plt.subplot(1,2,1)
# sns.boxplot(y=df["music_genre"], color="red")
# plt.subplot(1,2,2)
# sns.boxplot(y=df["friend_qualities"])
# plt.show()

km = KMeans(n_clusters=5)
clusters = km.fit_predict(df.iloc[:,1:])
df["label"] = clusters

 
# fig = plt.figure(figsize=(20,10))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df.columns[0][df.label == 0], df["year"][df.label == 0], df["humour"][df.label == 0], c='blue', s=60)
# ax.scatter(df.major[df.label == 1], df["year"][df.label == 1], df["humour"][df.label == 1], c='red', s=60)
# ax.scatter(df.major[df.label == 2], df["year"][df.label == 2], df["humour"][df.label == 2], c='green', s=60)
# ax.scatter(df.major[df.label == 3], df["year"][df.label == 3], df["humour"][df.label == 3], c='orange', s=60)
# ax.scatter(df.major[df.label == 4], df["year"][df.label == 4], df["humour"][df.label == 4], c='purple', s=60)
# ax.view_init(30, 185)
# plt.xlabel("major")
# plt.ylabel("year")
# ax.set_zlabel("humour")
# plt.show()

X = df[["personal_qualities","friend_qualities"]]
#Visualise data points
# plt.scatter(X["personal_qualities"],X["humour"],c='black')
# plt.xlabel('humour')
# plt.ylabel('personal_qualities')

K=5

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["personal_qualities"],X["friend_qualities"],c='black')
plt.scatter(Centroids["personal_qualities"],Centroids["friend_qualities"],c='red')
plt.xlabel('personal_qualities')
plt.ylabel('friend_qualities')
plt.show()

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
        diff = (Centroids_new['friend_qualities'] - Centroids['friend_qualities']).sum() + (Centroids_new['personal_qualities'] - Centroids['personal_qualities']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["friend_qualities","personal_qualities"]]

color=['blue','green','cyan', 'purple', 'black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["personal_qualities"],data["friend_qualities"],c=color[k])
plt.scatter(Centroids["personal_qualities"],Centroids["friend_qualities"],c='red')
plt.xlabel('personal_qualities')
plt.ylabel('friend_qualities')
plt.show()