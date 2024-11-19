import sys, os
import numpy as np
import pandas as pd
import math

from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QWidget, QVBoxLayout, QListWidgetItem
from PyQt6 import uic
from PyQt6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from MixtureModelAlgorithm import EM1, EM2, EM3  # Import from the original script
from BlinkExtractionAlgorithm import Cluster2d1d
from LocalPrecisionAlgorithm import Loc_Acc

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pyvista as pv

os.environ["TRAITSUI_TOOLKIT"] = "qt"
os.environ["ETS_TOOLKIT"] = "qt"
print("Initializing UI")


def update_plot_pyvista(df, sphere_radius=20):
    # Extract X, Y, and time frame positions
    x_positions = df.iloc[:, 0].to_numpy()
    y_positions = df.iloc[:, 1].to_numpy()
    time_frame = df.iloc[:, 2].to_numpy()

    # Create a PolyData object for the points
    points = pv.PolyData(np.column_stack((x_positions, y_positions, time_frame)))

    # Define a sphere for glyphing
    sphere = pv.Sphere(radius=sphere_radius)

    # Use glyphs to duplicate the sphere at each point's position
    glyphs = points.glyph(scale=False, geom=sphere)

    # Create a plotter and add the glyphs
    plotter = pv.Plotter()
    plotter.add_mesh(glyphs, color="blue")  # Adjust color as needed

    labels = dict(zlabel='Time (frame)', xlabel='X position (nm)', ylabel='Y position (nm)')
    plotter.show_grid(**labels)
    plotter.add_axes(**labels)
    plotter.camera_position = 'xy'  # View from top-down perspective
    plotter.show()


def visualize_spatial_clusters_pyvista(all_temporal_clusters, df, sphere_radius=20):
    plotter = pv.Plotter()

    for _, cluster in enumerate(all_temporal_clusters):
        x_coords = []
        y_coords = []
        z_coords = []

        for temporal_cluster in cluster:
            for index, time_frame in temporal_cluster:
                x_coords.append(df.iloc[index, 0])
                y_coords.append(df.iloc[index, 1])
                z_coords.append(time_frame)

        # Create a point cloud
        points = pv.PolyData(np.column_stack((x_coords, y_coords, z_coords)))

        # Create a sphere glyph
        sphere = pv.Sphere(radius=sphere_radius)
        glyphs = points.glyph(scale=False, geom=sphere)

        # Assign a unique color to the glyph
        color = np.random.rand(3)

        # Add the glyph to the plotter
        plotter.add_mesh(glyphs, color=color)

    labels = dict(zlabel='Time (frame)', xlabel='X position (nm)', ylabel='Y position (nm)')
    plotter.show_grid(**labels)
    plotter.add_axes(**labels)
    # light = pv.Light(position=(10, 10, 10))
    # light.diffuse_color = 1.0, 0.0, 0.0
    # plotter.add_light(light)
    # plotter.enable_ssao()
    plotter.camera_position = 'xy'  # View from top-down perspective
    plotter.show()

def visualize_temporal_clusters_pyvista(all_temporal_clusters, df, sphere_radius=20):
    plotter = pv.Plotter()

    for _, cluster in enumerate(all_temporal_clusters):

        for temporal_cluster in cluster:
            x_coords = []
            y_coords = []
            z_coords = []
            for index, time_frame in temporal_cluster:
                x_coords.append(df.iloc[index, 0])
                y_coords.append(df.iloc[index, 1])
                z_coords.append(time_frame)

            # Create a point cloud
            points = pv.PolyData(np.column_stack((x_coords, y_coords, z_coords)))

            # Create a sphere glyph
            sphere = pv.Sphere(radius=sphere_radius)
            glyphs = points.glyph(scale=False, geom=sphere)

            # Assign a unique color to the glyph
            color = np.random.rand(3)

            # Add the glyph to the plotter
            plotter.add_mesh(glyphs, color=color)
    
    labels = dict(zlabel='Time (frame)', xlabel='X position (nm)', ylabel='Y position (nm)')
    plotter.show_grid(**labels)
    plotter.add_axes(**labels)
    plotter.camera_position = 'xy'  # View from top-down perspective
    plotter.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        ui_path = self.resource_path("main.ui")

        uic.loadUi(ui_path, self)  # Load the UI file

        self.initialize_connections()

        self.inputLambda.setText("0")
        self.inputTheta.setText("1")

        self.preprocessing_clicked(None)

        self.blinking_data = None
        self.blinking_data_imported = False
        self.localization_data = None
        self.localization_data_imported = False
        self.lab_ineff = False
        self.analyzer = None

        self.replicates = 1
        self.subset_factor = 1
        self.local_precision = -1

        self.initialize_stoichiometry_graph()
        self.initialize_blinking_graph()

    def initialize_connections(self):
        # Connect the buttons to their respective functions
        self.runEMButton.clicked.connect(self.run_replicates)
        self.runExtractionButton.clicked.connect(self.run_blink_extraction)
        self.graphButton.clicked.connect(self.choose_graph)

        # Connect Menu items to their functions
        self.actionLoadBlinking.triggered.connect(self.load_blinking)
        self.actionLoadLocalization.triggered.connect(self.load_localization)
        self.actionGraph_Dataset.triggered.connect(self.plot_dataset)
        self.radioEM1.clicked.connect(self.set_default_pi)
        self.radioEM2.clicked.connect(self.set_default_pi)
        self.radioEM3.clicked.connect(self.set_default_pi)
        self.graph2dButton.clicked.connect(self.graph_2d_gaussian)

        self.preprocessingSwitch.mousePressEvent = self.preprocessing_clicked
        self.stoichiometrySwitch.mousePressEvent = self.stoichiometry_clicked

    def initialize_stoichiometry_graph(self):
        plt.rc('font', family='Calibri')

        # Create a Matplotlib figure and axes
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        self.bars = self.ax.bar(range(3), [0, 0, 0], color='gray', edgecolor='black', width=0.5)        # Set the X-axis labels and Y-axis limits
        # Set the X-axis labels
        self.ax.set_xticks(range(3))
        self.ax.set_xticklabels(['Monomer', 'Dimer', 'Trimer'])
        self.ax.set_ylim(0, 1)  # Set the y-axis limits to 0-100
        self.ax.set_ylabel("Distribution")
        # Reduce font size
        self.ax.tick_params(axis='both', labelsize=9)

        # Create a canvas and embed it in the graphWidget
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.graphWidget.layout().addWidget(self.canvas)

    def initialize_blinking_graph(self):

        # Create a Matplotlib figure and axes
        self.fig2, self.ax2 = plt.subplots(figsize=(10, 6))
        
        self.ax2.hist([], bins='auto', edgecolor='white')

        self.ax2.set_xlabel("Number of Blinks")
        self.ax2.set_ylabel("Frequency")
        # Reduce font size
        self.ax2.tick_params(axis='both', labelsize=9)

        # Create a canvas and embed it in the graphWidget
        self.canvas2 = FigureCanvasQTAgg(self.fig2)
        self.blinkGraph.layout().addWidget(self.canvas2)

    def resource_path(self, relative_path):
        """ Get the absolute path to the resource, works in development and after PyInstaller packaging """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def preprocessing_clicked(self, event):
        self.preprocessingSwitch.setStyleSheet("color: black;")
        self.stoichiometrySwitch.setStyleSheet("color: gray;")
        self.confStack.setCurrentIndex(1)
        self.resultStack.setCurrentIndex(1)
        self.graphStack.setCurrentIndex(1)

    def stoichiometry_clicked(self, event):
        self.preprocessingSwitch.setStyleSheet("color: gray;")
        self.stoichiometrySwitch.setStyleSheet("color: black;")
        self.confStack.setCurrentIndex(0)
        self.resultStack.setCurrentIndex(0)
        self.graphStack.setCurrentIndex(0)

    def load_blinking(self):
        # Open a file dialog to select the CSV file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Data File", "", "CSV Files (*.csv)")
        
        if file_path:
            # Display the file path in the UI
            self.blinkingFilePathLabel.setText(f"Loaded Blinking Dataset: {file_path}")

            # Load the CSV file using numpy
            self.blinking_data = np.genfromtxt(file_path, delimiter=",")
            self.blinking_data_imported = True
            self.stoichiometry_clicked(None)
        elif self.blinking_data_imported == True:
            pass
        else:
            self.blinkingFilePathLabel.setText("No file loaded")

    def load_localization(self):
        # Open a file dialog to select the CSV file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Data File", "", "Text Files (*.txt)")
        
        if file_path:
            # Display the file path in the UI
            self.localizationFilePathLabel.setText(f"Loaded Localization Dataset: {file_path}")

            # Load the CSV file using numpy
            self.localization_data = pd.read_csv(file_path, delimiter=' ', header=1)

            self.localization_data_imported = True
            self.preprocessing_clicked(None)
            p, e = Loc_Acc(self.localization_data)

            self.local_precision = p
            item = QListWidgetItem(f"{p:.2f}±({e:.2f})")
            self.valueListWidget.insertItem(1, item)

        elif self.localization_data_imported == True:
            pass
        else:
            self.localizationFilePathLabel.setText("No file loaded")

    def set_default_pi(self):
        if self.radioEM1.isChecked():
            self.inputPi.setText("0")
        elif self.radioEM2.isChecked():
            self.inputPi.setText("0,0")
        else:
            self.inputPi.setText("0,0,0")
    
    def run_blink_extraction(self):
        if not self.localization_data_imported:
            self.show_popup("Missing data", "Please load a localization file before running the extraction")
            return

        self.analyzer = Cluster2d1d(self.localization_data)
        self.analyzer.epsilon = int(self.epsInput.text())
        self.analyzer.min_sample = int(self.minSampleInput.text())
        self.analyzer.proximity = int(self.proxInput.text())
        self.analyzer.extract_features()
        self.analyzer.perform_dbscan()
        self.analyzer.get_all_temporal_clusters()
        blinking_data = self.analyzer.get_blinking_data()
        self.display_blinking_data(blinking_data)
        self.plot_blinking(blinking_data)

    def choose_graph(self):
        if self.radioOriginal.isChecked():
            if self.localization_data_imported:
                update_plot_pyvista(self.localization_data)
        elif self.radioSpatial.isChecked():
            if self.analyzer:
                visualize_spatial_clusters_pyvista(self.analyzer.all_temporal_clusters, self.localization_data)
        else:
            if self.analyzer:
                visualize_temporal_clusters_pyvista(self.analyzer.all_temporal_clusters, self.localization_data)

    def run_replicates(self):
        # Ensure data is loaded
        if self.blinking_data is None:
            # self.resultDisplay.setText("Please load a data file before running the algorithm")
            self.show_popup("Missing data", "Please load a data file before running the algorithm")
            return

        # Get user input for pi and lambda
        pi_input = self.inputPi.text()
        lambda_input = self.inputLambda.text()
        theta_input = self.inputTheta.text()

        self.replicates = int(self.replicatesInput.text())
        self.subset_factor = float(self.subsetSizeInput.text())

        # Process the input (convert string to list and float)
        try:
            pi = [float(x) for x in pi_input.split(',')] if pi_input else None
        except ValueError:
            self.show_popup("Invalid input", "Please enter the values in a comma-separated format")
            return

        lam = float(lambda_input) if lambda_input else None
        theta = float(theta_input) if theta_input else None

        self.lab_ineff = True if theta else False

        m, d, t = "M", "M/D", "M/D/T"
        
        # Select EMX based on user selection
        if self.radioEM1.isChecked():
            model = m
        elif self.radioEM2.isChecked():
            model = d
        elif self.radioEM3.isChecked():
            model = t
        else:
            # self.resultDisplay.setText("Please select an algorithm")
            self.show_popup("Missing algorithm", "Please select an algorithm")
            return
        
        pi_replicates, lam_replicates, aic_replicates = self.get_replicates(model, theta)

        transposed_pi = list(zip(*pi_replicates))
        transposed_lam = lam_replicates
        transposed_aic = aic_replicates

        print(transposed_pi, transposed_aic, transposed_lam)

        # Calculate the mean for each element using NumPy
        pi_means = [np.mean(sublist) for sublist in transposed_pi]
        pi_stds = [np.std(sublist) for sublist in transposed_pi]

        lam_means = np.mean(transposed_lam)

        aic_means = np.mean(transposed_aic)

        lam_std = np.std(transposed_lam)

        self.display_results(lam_means, pi_means, aic_means, pi_stds, lam_std, model)
        self.plot_result_with_error(pi_means, pi_stds, model)

    def display_blinking_data(self, data):
        text_to_display = "\n".join(str(item) for item in data)
        self.blinkListDisplay.setText(text_to_display)

    def display_results(self, lam, pi, AIC, pi_std, lam_std, model):
        
        if not all(0 <= value <= 1 for value in pi if value is not None):
            QMessageBox.warning(self, "Unphysical Values Predicted", "Predicted distribution values are outside the expected range (0-1). This may indicate an issue with the data or chosen parameters.")

        # Display the results in the QTextEdit widget
        row_position = 0
        self.tableWidget.insertRow(row_position)

        # Add Result Values to the new row
        item = QTableWidgetItem(str(round(pi[0]*100, 1)) + "(±" + str(round(pi_std[0]*100, 2)) + ")")
        # Make the item non-editable
        # item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.tableWidget.setItem(row_position, 0, item)

        if model == "M":
            item = QTableWidgetItem(str("N/A"))            
        else:
            item = QTableWidgetItem(str(round(pi[1]*100, 1)) + "(±" + str(round(pi_std[1]*100, 2)) + ")")
        self.tableWidget.setItem(row_position, 1, item)

        if model != "M/D/T":
            item = QTableWidgetItem(str("N/A"))
        else:
            item = QTableWidgetItem(str(round(pi[2]*100, 1)) + "(±" + str(round(pi_std[2]*100, 2)) + ")")
        self.tableWidget.setItem(row_position, 2, item)

        item = QTableWidgetItem(str(round(lam, 2)) + "(±" + str(round(lam_std, 2)) + ")")
        self.tableWidget.setItem(row_position, 3, item)

        item = QTableWidgetItem(str(round(AIC, 2)))
        self.tableWidget.setItem(row_position, 4, item)

        item = QTableWidgetItem(model)
        self.tableWidget.setItem(row_position, 5, item)

    def plot_result_with_error(self, values, std, model):
        """
        Plots the given values as a bar graph.

        Args:
            values (list): A list of three values to plot.
        """

        # Clear the existing plot
        self.ax.clear()

        # Create new bars
        self.bars = self.ax.bar(range(3), [0, 0, 0], color='gray', edgecolor='black', width=0.5)        # Set the X-axis labels and Y-axis limits
        self.ax.set_xticks(range(3))
        self.ax.set_xticklabels(['Monomer', 'Dimer', 'Trimer'])
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Distribution")
        # Reduce font size
        self.ax.tick_params(axis='both', labelsize=9)

        for bar, value in zip(self.bars, values):
            bar.set_height(value)

        self.ax.errorbar(range(len(std)), values, yerr=std, fmt='none', 
                     ecolor='black', capsize=5, capthick=2)

        self.canvas.draw()

    def show_popup(self, title, message):
        QMessageBox.information(self, title, message)

    def plot_blinking(self,blinking_data):

        # Sort the blinking data
        sorted_counts = sorted(blinking_data)

        self.ax2.clear()
        # Plot the distribution of blinking events
        self.ax2.hist(sorted_counts, bins=max(sorted_counts))
        self.ax2.set_xlabel("Number of Blinks")
        self.ax2.set_ylabel("Frequency")
        # Save the sorted counts to a CSV file
        np.savetxt("exported_data.csv", sorted_counts, delimiter=",")
        self.canvas2.draw()


    def plot_dataset(self):
        
        if not self.blinking_data_imported:
            # self.resultDisplay.setText("Please import the data file before plotting")
            self.show_popup("Missing data", "Please import the data file before plotting")

            return

        self.ax.clear()

        # Plot the data
        self.ax.plot(sorted(self.blinking_data))
        # Add labels and a title
        self.ax.set_xlabel("Dye (Sorted)")
        self.ax.set_ylabel("Number of Blinks")

        self.canvas.draw()

    def graph_2d_gaussian(self):
        pass

    def get_replicates(self, model, theta):
        bootstrapped_data = self.bootstrap_dataset(self.replicates, self.subset_factor)
        pi_replicates = [] 
        lam_replicates = []
        aic_replicates = []
        print(len(bootstrapped_data))

        for dataset in bootstrapped_data:
            if model == "M":
                em1 = EM1(dataset)
                em1.initialize()
                em1.run()
                pi_replicates.append(em1.pi)
                lam_replicates.append(em1.lam)
                aic_replicates.append(em1.AIC)
            elif model == "M/D":
                em2 = EM2(dataset)
                em2.initialize()
                em2.run()
                if self.lab_ineff:
                    em2.theta = theta
                    em2.apply_lab_ineff()
                pi_replicates.append(em2.pi)
                lam_replicates.append(em2.lam)
                aic_replicates.append(em2.AIC)
            else:
                em3 = EM3(dataset)
                em3.initialize()
                em3.run()
                if self.lab_ineff:
                    em3.theta = theta
                    em3.apply_lab_ineff()
                pi_replicates.append(em3.pi)
                lam_replicates.append(em3.lam)
                aic_replicates.append(em3.AIC)
        
        return pi_replicates, lam_replicates, aic_replicates


    def bootstrap_dataset(self, replicates, size_fraction):
        print(math.floor(len(self.blinking_data)*size_fraction))
        return [np.random.choice(self.blinking_data, size=math.floor(len(self.blinking_data)*size_fraction), replace=True) for _ in range(replicates)]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
