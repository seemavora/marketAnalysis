# import matplotlib.pyplot as plt 
# from sklearn.datasets.samples_generator import make_blobs 
# from sklearn.cluster import Birch 

# # Generating 600 samples using make_blobs 
# dataset, clusters = make_blobs(n_samples = 600, centers = 8, cluster_std = 0.75, random_state = 0) 
  
# # Creating the BIRCH clustering model 
# model = Birch(branching_factor = 50, n_clusters = None, threshold = 1.5) 
  
# # Fit the data (Training) 
# model.fit(dataset) 
  
# # Predict the same data 
# pred = model.predict(dataset) 
  
# # Creating a scatter plot 
# plt.scatter(dataset[:, 0], dataset[:, 1], c = pred, cmap = 'rainbow', alpha = 0.7, edgecolors = 'b') 
# plt.show()

import numpy as np
import argparse
import csv
from sklearn.cluster import Birch
import matplotlib.pyplot as plt

import pandas as pd
import chart_studio.plotly
# import chart_studio.graph_objs as go
import seaborn as sns

from typing import  Tuple, Dict, List

def load_data(file_name) -> List[List]:
	print("--->Loading csv file")

	with open("oneEncodedDiscord.csv") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=",")
		line_count = 0
		data = []

		for line in csv_reader:
			if line_count == 0:
				print(f'Column names: [{", ".join(line)}]')
			else:
				data.append(line)
			line_count += 1 

	print(f'Loaded {line_count} records')
	return data


def compute_clusters(data: List) -> np.ndarray:
	print("--->Computing clusters")
	birch = Birch(
		branching_factor=50,
		n_clusters=3,
		threshold=0.3,
		copy=True,
		compute_labels=True
	)

	birch.fit(data)
	predictions = np.array(birch.predict(data))
	print("Donee")
	print("predictions" ,predictions)

	return predictions

df = pd.read_csv("oneEncodedDiscord.csv")
def show_results(data: np.ndarray, labels: np.ndarray, plot_handler = "seaborn") -> None:
	labels = np.reshape(labels, (1, labels.size))
	data = np.concatenate((data, labels.T), axis=1)
    
	# Seaborn plot
	# if plot_handler == "seaborn":
	# 	print("SEABORN:")
	# 	facet = sns.lmplot(
	# 		x="music_genre", 
	# 		y="personal_qualities", 
	# 		fit_reg=False, 
	# 		legend=True, 
	# 		legend_out=True, 
    #         data = df
	# 	)

	facet = sns.lmplot(
			x="music_genre", 
			y="personal_qualities", 
			fit_reg=False, 
			legend=True, 
			legend_out=True, 
            data = df
	)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	scatter = ax.scatter(data[:,0], data[:, 1], c=data[:, 2], s=50)
	ax.set_title("Clusters")
	ax.set_xlabel("Music genre")
	ax.set_ylabel("Personal_qualities")
	plt.colorbar(scatter)
	plt.show()
	# Pure matplotlib plot
	# if plot_handler == "matplotlib":
	# 	print("MATPLOTLIB:")
	# 	fig = plt.figure()
	# 	ax = fig.add_subplot(111)
	# 	scatter = ax.scatter(data[:,0], data[:, 1], c=data[:, 2], s=50)
	# 	ax.set_title("Clusters")
	# 	ax.set_xlabel("Music genre")
	# 	ax.set_ylabel("Personal_qualities")
	# 	plt.colorbar(scatter)
	# plt.show()


def show_data_corelation(data=None, csv_file_name=None):
	data_set = None
	if csv_file_name is None:
		cor = np.corrcoef(data)
		print("Corelation matrix:")
		print(cor)
	else:
		data_set = pd.read_csv(csv_file_name)
		print("Else Corelation matrix:")
		print(data_set.describe())
		data_set = data_set[["music_genre", "personal_qualities", "friend_qualities"]]
		cor = data_set.corr()
	sns.heatmap(cor, square=True)
	plt.show()
	return data_set


def main(args) -> None:
	data = load_data("oneEncodedDiscord.csv")
	print("data", data)
	filtered_data = np.array([[item[3], item[4]] for item in data])

	data_set = None #Alternative data loaded using pandas
	# data_set = show_data_corelation(csv_file_name="oneEncodedDiscord.csv")

	filtered_data = np.array(filtered_data).astype(np.float64)
	print(filtered_data)
	labels = compute_clusters(filtered_data) 
	show_results(filtered_data, labels, args.plot_handler)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Do some clustering")
    # data_file = "data/cleanData/oneEncodedDiscord.csv"
	# parser.add_argument("--data-file", type=str, default="data/cleanData/oneEncodedDiscord.csv", help="dataset file name")
	parser.add_argument("--describe", type=bool, default=False, help="describe the dataset")
	parser.add_argument("--plot-handler", type=str, default="seaborn", help="what library to use for data visualisation")
	args = parser.parse_args()
	main(args)