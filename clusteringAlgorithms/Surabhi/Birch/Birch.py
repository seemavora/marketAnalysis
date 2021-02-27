import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()
import csv
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

df = pd.read_csv('../../data/cleanData/OHE_PA.csv')
df2 = pd.read_csv('../../data/cleanData/oneEncodedDiscord.csv')

# ~~~~~ Loading data from csv separating based on commas ~~~~~
data = []
with open('../../data/cleanData/OHE_PA.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=",")
		line_count = 0

		# Line 0 are columns (just being printed),
		# and the real data after is being appended to a list called data
		for line in csv_reader:
			if line_count == 0:
				print(f'Column names: [{", ".join(line)}]')
			else:
				data.append(line)
			line_count += 1 
# print(f'Loaded {line_count} records')
# print('data', data)

# ~~~~~ Filtering data and choosing which columns to plot ~~~~~
# For friends and personal qualities
filtered_data = np.array([[item[2], item[3]] for item in data])
filtered_data = np.array(filtered_data).astype(np.float64)
# print('filtered data', filtered_data)

# ~~~~~ Creating Birch object and calculating predictions ~~~~~
birch = Birch(
		branching_factor=50,
		n_clusters=5,
		threshold=0.3,
		copy=True,
		compute_labels=True
	)

birch.fit(filtered_data)
predictions = np.array(birch.predict(filtered_data))
print("predictions", predictions)

# ~~~~~ Unclustered scatter plot ~~~~~
facet = sns.lmplot(
		x="Business", 
		y="Sciences", 
		fit_reg=False, 
		legend=True, 
		legend_out=True, 
        data = df
)

# ~~~~~ Creating clustered plot ~~~~~
labels = np.reshape(predictions, (1, predictions.size))
fdata = np.concatenate((filtered_data, labels.T), axis=1)
print('fdata---', fdata)

# fig = plt.figure()
# # ax = fig.add_subplot(1, 1, 1)
# ax = fig.add_subplot(111)
# colors = fdata[:, 2]
# scatter = ax.scatter(fdata[:,0], fdata[:, 1], c=colors, s=50)
# ax.set_title("Clusters")
# ax.set_xlabel("friends_qualities")
# ax.set_ylabel("personal_qualities")
# # plt.colorbar(scatter)
# plt.show()

sns.set_style('whitegrid')
sns.countplot(x=df.index, hue='Business', data=df,palette='RdBu_r')
plt.show()

# random data
x = np.random.randint(0,1+1)

fig, ax = plt.subplots()

# define the colors
cmap = mpl.colors.ListedColormap(['r', 'k'])

# create a normalize object the describes the limits of
# each color
bounds = [0., 0.5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# plot it
ax.imshow(x, interpolation='none', cmap=cmap, norm=norm)