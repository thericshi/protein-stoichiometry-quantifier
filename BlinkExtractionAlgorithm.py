import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import random

class Cluster2d1d:
    def __init__(self, df):
        self.data = df
        self.clusters_2d = None
        self.time_clusters = None
        self.all_temporal_clusters = []

    def extract_features(self):
        if self.data is None:
            print("Data not loaded.")
            return
        self.x_positions = self.data.iloc[:, 0]
        self.y_positions = self.data.iloc[:, 1]
        self.time_frame = self.data.iloc[:, 2]
        self.positions = np.vstack((self.x_positions, self.y_positions)).T

    def perform_dbscan(self, eps=50, min_samples=8):
        self.dbscan_2d = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters_2d = self.dbscan_2d.fit_predict(self.positions)

    def visualize_clusters(self):
        if self.clusters_2d is None:
            print("Clusters not found. Please perform_dbscan() first.")
            return

        unique_clusters = np.unique(self.clusters_2d)

        random.seed(42)
        cluster_colors = {cluster: [random.random(), random.random(), random.random()] for cluster in unique_clusters}

        plt.figure(figsize=(10, 6))
        for cluster in unique_clusters:
            mask = (self.clusters_2d == cluster)
            plt.scatter(self.x_positions[mask], self.y_positions[mask], color=[cluster_colors[cluster]],
                        label=f'Cluster {cluster}' if cluster != -1 else 'Noise', s=50)

        plt.title('DBSCAN Clustering of Positions')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.ylim(26500, 0)  # Lower limit is 0, upper limit is 25000
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_temporal_clusters(self, series, proximity=2):
        if self.time_frame is None:
            print("Time frame data not available. Please extract_features() first.")
            return

        temporal_clusters = []
        current_subseries = []

        for index, value in series.items():
            if not current_subseries:
                current_subseries.append((index, value))
            else:
                _, last_value = current_subseries[-1]
                if value - last_value <= proximity:
                    current_subseries.append((index, value))
                else:
                    temporal_clusters.append(current_subseries)
                    current_subseries = [(index, value)]
        if current_subseries:
            temporal_clusters.append(current_subseries)

        return temporal_clusters

    def get_all_temporal_clusters(self):
        self.time_clusters = []
        for cluster in np.unique(self.clusters_2d):
            cluster_data = self.time_frame[self.clusters_2d == cluster]
            self.time_clusters.append(cluster_data)
            self.all_temporal_clusters.append(self.get_temporal_clusters(cluster_data, 2))

    def get_blinking_data(self):
        if self.time_clusters is None:
            print("Temporal clusters not found. Please get_temporal_clusters() first.")
            return

        blinking_data = []
        for spatial_cluster in self.all_temporal_clusters:
            for temporal_cluster in spatial_cluster:
                blinking_data.append(len(temporal_cluster))
        return blinking_data

    def plot_blinking(self):
        blinking_data = self.get_blinking_data()
        # Sort the blinking data
        sorted_counts = sorted(blinking_data)

        # Plot the distribution of blinking events
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_counts)), sorted_counts)
        plt.title("Distribution of Blinking Events")
        plt.xlabel("Cluster Index")
        plt.ylabel("Number of Blinks")
        plt.show()

        # Save the sorted counts to a CSV file
        np.savetxt("exported_data.csv", sorted_counts, delimiter=",")

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv('Total_Localization_file_initON_Spatial_Radius_2_pixel_init_OFF.txt', delimiter=' ', header=1)
    analyzer = Cluster2d1d(df)
    analyzer.extract_features()
    analyzer.perform_dbscan()
    # analyzer.visualize_clusters()
    analyzer.get_all_temporal_clusters()
    analyzer.plot_blinking()