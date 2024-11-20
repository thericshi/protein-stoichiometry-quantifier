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

        self.epsilon = 50
        self.min_sample = 8
        self.proximity = 2

    def extract_features(self):
        if self.data is None:
            print("Data not loaded.")
            return
        self.x_positions = self.data.iloc[:, 0]
        self.y_positions = self.data.iloc[:, 1]
        self.time_frame = self.data.iloc[:, 2]
        self.positions = np.vstack((self.x_positions, self.y_positions)).T

    def perform_dbscan(self):
        self.dbscan_2d = DBSCAN(eps=self.epsilon, min_samples=self.min_sample)
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

    def get_temporal_clusters(self, series):
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
                if value - last_value <= self.proximity:
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
            self.all_temporal_clusters.append(self.get_temporal_clusters(cluster_data))

    def get_blinking_data(self):
        if self.time_clusters is None:
            print("Temporal clusters not found. Please get_temporal_clusters() first.")
            return

        blinking_data = []
        for spatial_cluster in self.all_temporal_clusters:
            blinking_data.append(len(spatial_cluster))
        return blinking_data

    def plot_gaussian_clusters(self, sigma_nm, alpha_scale=0.5, intensity_scale=0.3, min_alpha=0.05, max_res=4096):
        """
        Plot clusters with each localization visualized as a Gaussian spot with physical size scaling.
        
        Parameters:
        - sigma_nm: Gaussian sigma in nanometers (physical units)
        - alpha_scale: Maximum alpha scaling factor (0-1) to control overall transparency
        - intensity_scale: Scaling factor for the brightness of each spot (0-1)
        - min_alpha: Minimum alpha value to ensure all points are visible
        """
        if self.clusters_2d is None:
            print("Clusters not found. Please perform_dbscan() first.")
            return

        def generate_distinct_colors(n):
            """Generate n visually distinct RGB colors using the golden ratio."""
            colors = []
            golden_ratio = 0.618033988749895
            hue = 0
            for _ in range(n):
                hue += golden_ratio
                hue %= 1.0
                h = hue * 6
                c = 0.8
                x = c * (1 - abs(h % 2 - 1))
                if h < 1: r, g, b = c, x, 0
                elif h < 2: r, g, b = x, c, 0
                elif h < 3: r, g, b = 0, c, x
                elif h < 4: r, g, b = 0, x, c
                elif h < 5: r, g, b = x, 0, c
                else: r, g, b = c, 0, x
                m = 0.2  # Brightness factor
                colors.append((r + m, g + m, b + m))
            return colors

        # Physical dimensions
        x_range_nm = self.x_positions.max() - self.x_positions.min()
        y_range_nm = self.y_positions.max() - self.y_positions.min()

        # Grid size
        pixels_per_sigma = 5
        required_pixels = max(x_range_nm, y_range_nm) / sigma_nm * pixels_per_sigma
        grid_size = int(2 ** np.ceil(np.log2(required_pixels)))
        grid_size = min(max(grid_size, 1024), max_res)
        pixel_size_nm = max(x_range_nm, y_range_nm) / grid_size
        sigma_pixels = sigma_nm / pixel_size_nm

        # Initialize image
        image = np.zeros((grid_size, grid_size, 4))
        x_scaled = ((self.x_positions - self.x_positions.min()) / x_range_nm * (grid_size - 1))
        y_scaled = ((self.y_positions - self.y_positions.min()) / y_range_nm * (grid_size - 1))

        # Gaussian kernel
        kernel_size = int(4 * sigma_pixels)
        x = np.linspace(-kernel_size, kernel_size, 2 * kernel_size + 1)
        y = np.linspace(-kernel_size, kernel_size, 2 * kernel_size + 1)
        xx, yy = np.meshgrid(x, y)
        gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_pixels**2))
        gaussian_kernel /= gaussian_kernel.max()

        # Normalize intensities
        intensities = self.data.iloc[:, 3].values
        q1, q99 = np.percentile(intensities, [1, 99])
        normalized_intensities = np.clip((intensities - q1) / (q99 - q1), 0, 1)
        normalized_intensities = np.power(normalized_intensities, 0.5)

        # Cluster colors
        unique_clusters = np.unique(self.clusters_2d)
        distinct_colors = generate_distinct_colors(len(unique_clusters))
        cluster_colors = {cluster: distinct_colors[i] for i, cluster in enumerate(unique_clusters)}

        # Plot each localization
        for x, y, intensity, cluster in zip(x_scaled, y_scaled, normalized_intensities, self.clusters_2d):
            x, y = int(x), int(y)
            alpha = np.clip(intensity * intensity_scale * alpha_scale + min_alpha, min_alpha, 1.0)
            base_color = np.array(cluster_colors[cluster])
            spot = np.zeros((2 * kernel_size + 1, 2 * kernel_size + 1, 4))
            spot[..., :3] = base_color
            spot[..., 3] = gaussian_kernel * alpha

            x_min = max(x - kernel_size, 0)
            x_max = min(x + kernel_size + 1, grid_size)
            y_min = max(y - kernel_size, 0)
            y_max = min(y + kernel_size + 1, grid_size)
            spot_x_min = kernel_size - (x - x_min)
            spot_x_max = spot_x_min + (x_max - x_min)
            spot_y_min = kernel_size - (y - y_min)
            spot_y_max = spot_y_min + (y_max - y_min)

            spot_section = spot[spot_x_min:spot_x_max, spot_y_min:spot_y_max]
            img_section = image[x_min:x_max, y_min:y_max]
            alpha_spot = spot_section[..., 3:4]
            alpha_img = img_section[..., 3:4]

            new_alpha = alpha_spot + alpha_img * (1 - alpha_spot)
            denominator = np.maximum(new_alpha, 1e-10)
            new_colors = ((spot_section[..., :3] * alpha_spot +
                        img_section[..., :3] * alpha_img * (1 - alpha_spot)) / denominator)

            img_section[..., :3] = new_colors
            img_section[..., 3:4] = new_alpha

        # Local plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, origin='lower',
                extent=[self.x_positions.min(), self.x_positions.max(),
                        self.y_positions.min(), self.y_positions.max()],
                interpolation='nearest')

        # Scale bar
        scale_bar_length = x_range_nm / 10
        scale_bar_length_rounded = float(f"{scale_bar_length:.2g}")
        x_pos = self.x_positions.min() + x_range_nm * 0.05
        y_pos = self.y_positions.min() + y_range_nm * 0.05
        ax.plot([x_pos, x_pos + scale_bar_length_rounded], [y_pos, y_pos], 'w-', linewidth=2)
        ax.text(x_pos + scale_bar_length_rounded / 2, y_pos - y_range_nm * 0.02,
                f'{scale_bar_length_rounded} nm', color='white', ha='center')

        ax.set_title(f"Gaussian-Blurred Localizations (σ = {sigma_nm:.1f} nm, pixel size = {pixel_size_nm:.2f} nm)")
        ax.set_xlabel("X Position (nm)")
        ax.set_ylabel("Y Position (nm)")

        fig.show()

        # Print resolution
        print(f"Image resolution: {grid_size}x{grid_size} pixels")
        print(f"Pixel size: {pixel_size_nm:.2f} nm")
        print(f"Gaussian σ: {sigma_nm:.1f} nm ({sigma_pixels:.1f} pixels)")

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv('Total_Localization_file_initON_Spatial_Radius_2_pixel_init_OFF.txt', delimiter=' ', header=1)
    analyzer = Cluster2d1d(df)
    analyzer.extract_features()
    analyzer.perform_dbscan()
    # analyzer.visualize_clusters()
    analyzer.get_all_temporal_clusters()
    analyzer.plot_blinking()