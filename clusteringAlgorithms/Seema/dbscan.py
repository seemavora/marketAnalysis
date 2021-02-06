# https://www.datacamp.com/community/tutorials/dbscan-macroscopic-investigation-python

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

data = pd.read_csv('../../data/cleanData/oneEncodedDiscord.csv')
data = data[['personal_qualities','friend_qualities']].to_numpy()
# data = pd.DataFrame(data, columns= ['personal_qualities','friend_qualities'])
data = data.astype("float32",copy = False)

stscaler = StandardScaler().fit(data)
data = stscaler.transform(data)

dbsc = DBSCAN(eps=.5, min_samples = 15).fit(data)

labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True

cmap = cm.get_cmap('Accent')
data = pd.DataFrame(data, columns= ['personal_qualities','friend_qualities'])
data.plot.scatter(x = 'personal_qualities',y = 'friend_qualities', cmap= cmap, colorbar = False)