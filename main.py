print("Initializing UI")

import sys, os
import numpy as np
import pandas as pd
import math

from PyQt6.QtWidgets import QDialog, QPushButton, QApplication, QLabel, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QWidget, QVBoxLayout, QListWidgetItem, QDockWidget, QStatusBar, QProgressBar, QHBoxLayout, QLineEdit, QTextBrowser

from PyQt6 import uic
from PyQt6.QtCore import QThread, QSize, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QDoubleValidator
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from MixtureModelAlgorithm import EM1, EM2, EM3  # Import from the original script
from BlinkExtractionAlgorithm import Cluster2d1d
from LocalPrecisionAlgorithm import Loc_Acc
from PyVistaPlotter import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


os.environ["TRAITSUI_TOOLKIT"] = "qt"
os.environ["ETS_TOOLKIT"] = "qt"


class HelpDialog(QDialog):
    def __init__(self):
        super(HelpDialog, self).__init__()

        self.setWindowTitle("User Guide")
        self.resize(QSize(650, 450))
        # Layout
        layout = QVBoxLayout()

        # Program information using QTextBrowser
        info_text_browser = QTextBrowser()

        # Read HTML content from user_guide_html.txt in the same directory as the script
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
            file_path = os.path.join(script_dir, "user_guide_html.txt")  # Construct the full file path

            with open(file_path, "r", encoding="utf-8") as file:
                html_content = file.read()
            info_text_browser.setHtml(html_content)
        except FileNotFoundError:
            info_text_browser.setHtml("""
                <h1>Error</h1>
                <p>User guide file (user_guide_html.txt) not found in the script's directory.</p>
            """)

        info_text_browser.setOpenExternalLinks(True)  # Allow clickable links

        layout.addWidget(info_text_browser)

        # OK button to close the dialog
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)


class AboutDialog(QDialog):
    def __init__(self):
        super(AboutDialog, self).__init__()

        self.setWindowTitle("About")

        # Layout
        layout = QVBoxLayout()

        # Program information
        info_label = QLabel(
            "Protein Stoichiometry Quantifier\n\n"
            "Date: 2024-11\n"
            "Developed by: Eric Shi in the Milstein Lab, University of Toronto\n"
            "This program utilizes algorithms developed by:\n"
            "Artittaya Boonkird, Daniel F Nino and Joshua N Milstein in the Milstein Lab: https://doi.org/10.1093/bioadv/vbab032 for the prediction of protein stoichiometry\n"
            "Ulrike Endesfelder, Sebastian Malkusch, Franziska Fricke and Mike Heilemann: https://pubmed.ncbi.nlm.nih.gov/24522395/ for the estimation of localization precision\n"
        )
        info_label.setOpenExternalLinks(True)  # Allow clickable links
        info_label.setWordWrap(True)

        layout.addWidget(info_label)

        # OK button to close the dialog
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)


class ModifyAttributesDialog(QDialog):
    def __init__(self, parent=None):
        super(ModifyAttributesDialog, self).__init__(parent)
        self.setWindowTitle("Modify Default Attributes")
        
        # Layout
        layout = QVBoxLayout()
        
        # Create input field for max_iter
        max_iter_layout = QHBoxLayout()
        max_iter_label = QLabel("Maximum Iterations:")
        self.max_iter_input = QLineEdit()
        self.max_iter_input.setText("50000")  # Default value
        max_iter_layout.addWidget(max_iter_label)
        max_iter_layout.addWidget(self.max_iter_input)
        layout.addLayout(max_iter_layout)
        
        # Add some spacing
        layout.addSpacing(20)
        
        # Add OK and Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def get_max_iter(self):
        return int(self.max_iter_input.text())


class DataHandler:
    def __init__(self, main_window):
        self.main_window = main_window  # Store a reference to the MainWindow
        self.blinking_data = None
        self.blinking_data_imported = False
        self.localization_data = None
        self.localization_data_imported = False
        self.local_precision = -1
        self.local_precision_error = None  # Store any error message

    def load_blinking(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self.main_window, "Open Data File", "", "CSV Files (*.csv)")

        if file_path:
            self.main_window.blinkingFilePathLabel.setText(f"Loaded Blinking Dataset: {file_path}")
            try:
                self.blinking_data = np.genfromtxt(file_path, delimiter=",")
                self.blinking_data_imported = True
                self.main_window.blinking_data = self.blinking_data 
                self.main_window.blinking_data_imported = self.blinking_data_imported
                self.main_window.stoichiometry_clicked(None)  # Switch tabs
            except Exception as e:
                self.main_window.blinkingFilePathLabel.setText(f"Error loading file: {e}")
                self.blinking_data_imported = False  # Set to False if loading fails
                self.blinking_data = None
                self.main_window.blinking_data = self.blinking_data # Update MainWindow variable
                self.main_window.blinking_data_imported = self.blinking_data_imported

        elif self.blinking_data_imported: # Keep the previous data if the user cancels the file dialog and data was already loaded
             pass
        else:
            self.main_window.blinkingFilePathLabel.setText("No file loaded")


    def load_localization(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self.main_window, "Open Data File", "", "Text Files (*.txt)")

        if file_path:
            try:
                # First try to read the file to validate format
                with open(file_path, 'r') as f:
                    # Skip header line
                    header = f.readline()
                    # Read first few lines to validate format
                    first_line = f.readline().strip()
                    if not first_line:
                        raise ValueError("File is empty after header")
                    
                    # Split the line and validate number of columns
                    columns = first_line.split()
                    if len(columns) < 4:
                        raise ValueError("File must have at least 4 columns")
                    
                    # Validate data types
                    try:
                        float(columns[0])  # First column should be float/int
                        float(columns[1])  # Second column should be float/int
                        int(columns[2])    # Third column should be integer
                        float(columns[3])  # Fourth column should be float/int
                    except ValueError as e:
                        raise ValueError("Invalid data types in columns. Expected: float/int, float/int, int, float/int")

                # If validation passes, load the file
                self.localization_data = pd.read_csv(file_path, delimiter=' ', header=1)
                self.localization_data_imported = True
                self.main_window.localization_data = self.localization_data 
                self.main_window.localization_data_imported = self.localization_data_imported
                self.main_window.localizationFilePathLabel.setText(f"Loaded Localization Dataset: {file_path}")
                # Enable the Extract Blinks button
                self.main_window.runExtractionButton.setEnabled(True)
                # Disable the Proceed button until extraction is done
                self.main_window.pushButton.setEnabled(False)

                try:
                    p, e = Loc_Acc(self.localization_data)
                    self.local_precision = p
                    self.local_precision_error = e
                    self.main_window.local_precision = self.local_precision

                    item = QListWidgetItem(f"{p:.2f}±({e:.2f})")
                    self.main_window.valueListWidget.insertItem(1, item)
                    self.main_window.preprocessing_clicked(None)  # Switch tabs
                except Exception as exc:
                    item = QListWidgetItem(f"Optimal parameter not found")
                    self.main_window.valueListWidget.insertItem(1, item)
                    self.main_window.preprocessing_clicked(None)  # Switch tabs

            except Exception as ex:
                self.main_window.show_popup("Invalid File Format", 
                    "The file must be space-separated with the following format:\n"
                    "- First column (x-position): float or integer\n"
                    "- Second column (y-position): float or integer\n"
                    "- Third column (frame number): integer\n"
                    "- Fourth column (intensity): float or integer\n"
                    f"\nError: {str(ex)}")
                self.localization_data_imported = False
                self.localization_data = None
                self.local_precision = -1
                self.local_precision_error = None
                self.main_window.localization_data = self.localization_data
                self.main_window.localization_data_imported = self.localization_data_imported
                self.main_window.local_precision = self.local_precision
                self.main_window.localizationFilePathLabel.setText("No file loaded")
                # Disable both buttons when data is invalid
                self.main_window.runExtractionButton.setEnabled(False)
                self.main_window.pushButton.setEnabled(False)

        elif self.localization_data_imported:  # Keep the previous data if the user cancels the file dialog and data was already loaded
            pass
        else:
            self.main_window.localizationFilePathLabel.setText("No file loaded")
            self.local_precision = -1  # Reset if no file is loaded
            self.local_precision_error = None
            self.main_window.local_precision = self.local_precision
            # Disable both buttons when no data is loaded
            self.main_window.runExtractionButton.setEnabled(False)
            self.main_window.pushButton.setEnabled(False)


class EMAlgorithmExecution(QThread):
    finished_signal = pyqtSignal(object, object, object, object, object, object)  # Signal for results
    progress_update = pyqtSignal(int)  # Signal for progress updates
    cancelled_signal = pyqtSignal()  # Signal for cancellation
    max_iter_error_signal = pyqtSignal(str, str)  # Signal for max iterations error (title, message)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.lab_ineff = False
        self.model = None
        self.is_cancelled = False  # Flag to track cancellation
        # Connect the max_iter_error_signal to the show_popup method
        self.max_iter_error_signal.connect(self.main_window.show_popup)

    def run(self):  # Override the run method (this is what the thread executes)
        theta_input = self.main_window.inputTheta.text()

        self.main_window.replicates = int(self.main_window.replicatesInput.text())
        self.main_window.subset_factor = float(self.main_window.subsetSizeInput.text())

        theta = float(theta_input) if theta_input else None

        self.lab_ineff = True if theta else False

        m, d, t = "M", "M/D", "M/D/T"

        if self.main_window.radioEM1.isChecked():
            self.model = m
        elif self.main_window.radioEM2.isChecked():
            self.model = d
        else:
            self.model = t

        pi_replicates, lam_replicates, aic_replicates = self._get_replicates(self.model, theta) # Pass self.model

        # If we got empty results (max iterations exceeded), emit empty results
        if not pi_replicates or not lam_replicates or not aic_replicates:
            self.finished_signal.emit(None, None, None, None, None, self.model)
            return

        transposed_pi = list(zip(*pi_replicates))
        transposed_lam = lam_replicates
        transposed_aic = aic_replicates

        pi_means = [np.mean(sublist) for sublist in transposed_pi]
        pi_stds = [np.std(sublist) for sublist in transposed_pi]

        lam_means = np.mean(transposed_lam)
        aic_means = np.mean(transposed_aic)
        lam_std = np.std(transposed_lam)

        # Emit the signal with the results:
        self.finished_signal.emit(lam_means, pi_means, aic_means, pi_stds, lam_std, self.model)

    def _get_replicates(self, model, theta):
        bootstrapped_data = self._bootstrap_dataset(self.main_window.replicates, self.main_window.subset_factor)
        pi_replicates = []
        lam_replicates = []
        aic_replicates = []
        progress = 0

        for i, dataset in enumerate(bootstrapped_data):
            if self.is_cancelled:  # Check cancellation flag in the loop
                self.cancelled_signal.emit()  # Emit cancellation signal
                return [], [], []  # Return empty lists to stop further processing
            try:
                if model == "M":
                    em1 = EM1(dataset)
                    em1.initialize()
                    em1.run(max_iter=self.main_window.max_iter)
                    pi_replicates.append(em1.pi)
                    lam_replicates.append(em1.lam)
                    aic_replicates.append(em1.AIC)
                elif model == "M/D":
                    em2 = EM2(dataset)
                    em2.initialize()
                    em2.run(max_iter=self.main_window.max_iter)
                    if self.lab_ineff:
                        em2.theta = theta
                        em2.apply_lab_ineff()
                    pi_replicates.append(em2.pi)
                    lam_replicates.append(em2.lam)
                    aic_replicates.append(em2.AIC)
                else:  # model == "M/D/T"
                    em3 = EM3(dataset)
                    em3.initialize()
                    em3.run(max_iter=self.main_window.max_iter)
                    if self.lab_ineff:
                        em3.theta = theta
                        em3.apply_lab_ineff()
                    pi_replicates.append(em3.pi)
                    lam_replicates.append(em3.lam)
                    aic_replicates.append(em3.AIC)
            except RuntimeError as e:
                if "Maximum iterations" in str(e):
                    self.max_iter_error_signal.emit(
                        "Maximum Iterations Exceeded",
                        f"The EM algorithm exceeded the maximum number of iterations ({self.main_window.max_iter}). This can be modified in the settings menu."
                    )
                    return [], [], []  # Return empty lists to stop further processing

            progress_percentage = int(round((i + 1) / len(bootstrapped_data) * 100))
            self.progress_update.emit(progress_percentage)  # Emit the progress update signal

        return pi_replicates, lam_replicates, aic_replicates

    def _bootstrap_dataset(self, replicates, size_fraction):
        return [np.random.choice(self.main_window.blinking_data, size=math.floor(len(self.main_window.blinking_data) * size_fraction), replace=False) for _ in range(replicates)]

    def cancel(self):  # Method to set the cancellation flag
        self.is_cancelled = True

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        ui_path = self.resource_path("main.ui")

        uic.loadUi(ui_path, self)  # Load the UI file

        positive_validator = QDoubleValidator()
        positive_validator.setRange(0.0, float('inf'))
        positive_validator.setDecimals(3)

        self.initialize_connections()

        self.inputTheta.setText("1")
        self.inputTheta.setValidator(positive_validator)

        self.preprocessing_clicked(None)

        # Initially disable buttons
        self.runExtractionButton.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.graphButton.setEnabled(False)
        self.graph2dButton.setEnabled(False)
        self.runEMButton.setEnabled(False)

        self.blinking_data = None
        self.blinking_data_imported = False
        self.localization_data = None
        self.localization_data_imported = False
        self.lab_ineff = False
        self.analyzer = None

        self.replicates = 1
        self.subset_factor = 1
        self.local_precision = -1
        self.max_iter = 50000  # Default value for max iterations

        self.initialize_stoichiometry_graph()
        self.initialize_blinking_graph()
        self.set_window_size()
        
        self.data_handler = DataHandler(self)  # Pass 'self' (the MainWindow instance)
        self.em_thread = EMAlgorithmExecution(self)
        self.em_thread.finished_signal.connect(self.handle_em_results)
        self.em_thread.started.connect(self.thread_started)
        self.em_thread.finished.connect(self.thread_finished)
        self.em_thread.progress_update.connect(self.update_progress_bar)
        self.em_thread.cancelled_signal.connect(self.algorithm_cancelled)


        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.progressBar = QProgressBar() 
        self.statusBar.addPermanentWidget(self.progressBar)  # Add it to the status bar
        self.progressBar.setMaximumHeight(10)  # Set maximum height to make it more compact
        self.progressBar.hide()

        self.cancel_button = QPushButton("Cancel", self.statusBar)
        self.statusBar.addPermanentWidget(self.cancel_button)
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self.cancel_em_algorithm)

    def show_help_dialog(self):
        """Display the Help dialog."""
        help_dialog = HelpDialog()
        help_dialog.exec()  

    def show_about_dialog(self):
        """Display the About dialog."""
        about_dialog = AboutDialog()
        about_dialog.exec()
    
    def set_window_size(self):
        # Get the primary screen's geometry
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Calculate the new dimensions (2/3 of the screen size)
        new_width = int(screen_width * 2 / 3)
        new_height = int(screen_height * 2 / 3)

        # Center the window and resize
        self.setGeometry(
            int((screen_width - new_width) / 2),
            int((screen_height - new_height) / 2),
            new_width,
            new_height,
        )

    def initialize_connections(self):
        # Connect the buttons to their respective functions
        self.runEMButton.clicked.connect(self.run_replicates)
        self.runExtractionButton.clicked.connect(self.run_blink_extraction)
        self.graphButton.clicked.connect(self.choose_graph)
        self.pushButton.clicked.connect(self.proceed_to_stoichiometry)  # Add connection for proceed button
        self.maxIterButton.clicked.connect(self.show_modify_attributes_dialog)

        # Connect Menu items to their functions
        self.actionLoadBlinking.triggered.connect(self.load_blinking)
        self.actionLoadLocalization.triggered.connect(self.load_localization)
        self.actionGraph_Dataset.triggered.connect(self.plot_dataset)
        self.actionAbout.triggered.connect(self.show_about_dialog)
        self.actionUserGuide.triggered.connect(self.show_help_dialog)

        self.graph2dButton.clicked.connect(self.graph_2d_gaussian)

        self.preprocessingSwitch.mousePressEvent = self.preprocessing_clicked
        self.stoichiometrySwitch.mousePressEvent = self.stoichiometry_clicked

    def initialize_stoichiometry_graph(self):
        plt.rc('font', family='Calibri')

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        self.bars = self.ax.bar(range(3), [0, 0, 0], color='gray', edgecolor='black', width=0.5)        # Set the X-axis labels and Y-axis limits
        self.ax.set_xticks(range(3))
        self.ax.set_xticklabels(['Monomer', 'Dimer', 'Trimer'])
        self.ax.set_ylim(0, 1)  # Set the y-axis limits to 0-100
        self.ax.set_ylabel("Distribution")
        self.ax.tick_params(axis='both', labelsize=9)

        # Create a canvas and embed it in the graphWidget
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.graphWidget.layout().addWidget(self.canvas)

    def initialize_blinking_graph(self):

        self.fig2, self.ax2 = plt.subplots(figsize=(10, 6))
        
        self.ax2.hist([], bins='auto', edgecolor='white')

        self.ax2.set_xlabel("Number of Blinks")
        self.ax2.set_ylabel("Frequency")
        self.ax2.tick_params(axis='both', labelsize=9)

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
        self.data_handler.load_blinking()
        # Enable the EM button if data was loaded successfully
        self.runEMButton.setEnabled(self.blinking_data_imported)

    def load_localization(self):
        self.data_handler.load_localization()
        # Enable/disable buttons based on data state
        self.runExtractionButton.setEnabled(self.localization_data_imported)
        # Both graph buttons should be disabled until extraction is done
        self.graphButton.setEnabled(False)
        self.graph2dButton.setEnabled(False)
        self.pushButton.setEnabled(False)  # Disabled until extraction is done

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
        
        # Enable all buttons after successful extraction
        self.pushButton.setEnabled(True)
        self.graphButton.setEnabled(True)
        self.graph2dButton.setEnabled(True)

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

    def display_blinking_data(self, data):
        text_to_display = "\n".join(str(item) for item in data)
        self.blinkListDisplay.setText(text_to_display)

    def display_results(self, lam, pi, AIC, pi_std, lam_std, model):
        
        if not all(0 <= value <= 1 for value in pi if value is not None):
            QMessageBox.warning(self, "Unphysical Values Predicted", "Predicted distribution values are outside the expected range (0-1). This may indicate an issue with the data or chosen parameters.")

        row_position = 0
        self.tableWidget.insertRow(row_position)

        # Add Result Values to the new row
        item = QTableWidgetItem(str(round(pi[0]*100, 1)))
        # Make the item non-editable
        # item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.tableWidget.setItem(row_position, 0, item)

        if model == "M":
            item = QTableWidgetItem(str("N/A"))            
        else:
            item = QTableWidgetItem(str(round(pi[1]*100, 1)))
        self.tableWidget.setItem(row_position, 1, item)

        if model != "M/D/T":
            item = QTableWidgetItem(str("N/A"))
        else:
            item = QTableWidgetItem(str(round(pi[2]*100, 1)))
        self.tableWidget.setItem(row_position, 2, item)

        item = QTableWidgetItem(str(round(lam, 2)))
        self.tableWidget.setItem(row_position, 3, item)

        item = QTableWidgetItem(str(round(AIC, 2)))
        self.tableWidget.setItem(row_position, 4, item)

        item = QTableWidgetItem(model)
        self.tableWidget.setItem(row_position, 5, item)

    def plot_stoichiometry(self, values, std, model):
        """
        Plots the given values as a bar graph.

        Args:
            values (list): A list of three values to plot.
        """

        self.ax.clear()

        self.bars = self.ax.bar(range(3), [0, 0, 0], color='gray', edgecolor='black', width=0.5)        # Set the X-axis labels and Y-axis limits
        self.ax.set_xticks(range(3))
        self.ax.set_xticklabels(['Monomer', 'Dimer', 'Trimer'])
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Distribution")
        self.ax.tick_params(axis='both', labelsize=9)

        for bar, value in zip(self.bars, values):
            bar.set_height(value)

        self.ax.errorbar(range(len(std)), values, yerr=std, fmt='none', 
                     ecolor='black', capsize=5, capthick=2)

        self.canvas.draw()

    def show_popup(self, title, message):
        QMessageBox.information(self, title, message)

    def plot_blinking(self,blinking_data):

        sorted_counts = sorted(blinking_data)

        self.ax2.clear()
        # Plot the distribution of blinking events
        self.ax2.hist(sorted_counts, bins=max(sorted_counts))
        self.ax2.set_xlabel("Number of Blinks")
        self.ax2.set_ylabel("Frequency")

        np.savetxt("exported_data.csv", sorted_counts, delimiter=",")
        self.canvas2.draw()

    def plot_dataset(self):
        
        if not self.blinking_data_imported:
            self.show_popup("Missing data", "Please import the data file before plotting")
            return

        self.ax.clear()

        self.ax.plot(sorted(self.blinking_data))
        self.ax.set_xlabel("Dye (Sorted)")
        self.ax.set_ylabel("Number of Blinks")

        self.canvas.draw()

    def graph_2d_gaussian(self):
        if self.analyzer:
            max_res = 8192
            alpha_scale = 0.8
 
            if self.radio2dClusters.isChecked():
                self.analyzer.plot_gaussian_clusters(self.local_precision, alpha_scale=alpha_scale, intensity_scale=0.3, min_alpha=0.05, max_res=max_res)
            elif self.radio2dPoints.isChecked():
                if not self.analyzer.all_temporal_clusters:
                    self.show_popup("No Data", "Please run the extraction algorithm first.")
                    return
                plot_2d_points_clusters(self.analyzer.all_temporal_clusters, self.localization_data)

    def run_replicates(self):
        if self.blinking_data is None:
            self.show_popup("Missing data", "Please load a data file before running the algorithm")
            return
        
        # Validate theta (labeling efficiency factor) if provided
        theta_input = self.inputTheta.text()
        if theta_input:
            try:
                theta = float(theta_input)
                if theta <= 0 or theta > 1:
                    self.show_popup("Invalid Input", 
                        "Labeling efficiency factor must be greater than 0 and less than or equal to 1.")
                    return
            except ValueError:
                self.show_popup("Invalid Input", 
                    "Labeling efficiency factor must be a valid number.")
                return
        
        # Validate subset size factor
        try:
            subset_factor = float(self.subsetSizeInput.text())
            if subset_factor <= 0 or subset_factor > 1:
                self.show_popup("Invalid Input", 
                    "Subset size factor must be greater than 0 and less than or equal to 1.")
                return
        except ValueError:
            self.show_popup("Invalid Input", 
                "Subset size factor must be a valid number.")
            return
        
        m, d, t = "M", "M/D", "M/D/T"

        if self.radioEM1.isChecked():
            self.model = m
        elif self.radioEM2.isChecked():
            self.model = d
        elif self.radioEM3.isChecked():
            self.model = t
        else:
            self.show_popup("Missing algorithm", "Please select an algorithm")
            return
            
        self.em_thread.start()
        self.progressBar.show()
        self.progressBar.setValue(0)
        self.cancel_button.show()

    def thread_started(self):
        self.runEMButton.setEnabled(False)

    def thread_finished(self):
        self.runEMButton.setEnabled(True)
        self.progressBar.hide()
        self.cancel_button.hide()

    def update_progress_bar(self, progress):
        self.progressBar.setValue(progress)

    def handle_em_results(self, lam_means, pi_means, aic_means, pi_stds, lam_std, model):
        if self.em_thread.is_cancelled:
            self.em_thread.is_cancelled = False
            return
            
        # Check if we have empty results (which happens when max iterations is exceeded)
        if not pi_means or not pi_stds:
            return  # Just return without trying to display or plot results
            
        self.display_results(lam_means, pi_means, aic_means, pi_stds, lam_std, model)
        self.plot_stoichiometry(pi_means, pi_stds, model)
        self.runEMButton.setEnabled(True)

    def algorithm_cancelled(self):
        self.runEMButton.setEnabled(True)
        self.progressBar.hide()
        self.cancel_button.hide()
        self.show_popup("Algorithm Cancelled", "The EM algorithm has been cancelled.")

    def cancel_em_algorithm(self):
        self.em_thread.cancel()

    def show_modify_attributes_dialog(self):
        """Display the Modify Attributes dialog."""
        dialog = ModifyAttributesDialog(self)
        dialog.max_iter_input.setText(str(self.max_iter))  # Set current value
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.max_iter = dialog.get_max_iter()
            self.show_popup("Settings Updated", f"Maximum iterations has been set to {self.max_iter}")

    def proceed_to_stoichiometry(self):
        """Switch to stoichiometry tab and transfer the extracted blink data."""
        if not self.analyzer or not self.analyzer.all_temporal_clusters:
            self.show_popup("No Data", "Please run the extraction algorithm first.")
            return
            
        # Get the blinking data
        blinking_data = self.analyzer.get_blinking_data()
        
        # Save the blinking data to a CSV file
        np.savetxt("exported_data.csv", blinking_data, delimiter=",")
        
        # Load the data into the blinking_data variable
        self.blinking_data = np.array(blinking_data)
        self.blinking_data_imported = True
        
        # Switch to stoichiometry tab
        self.stoichiometry_clicked(None)
        
        # Update the file path label
        self.blinkingFilePathLabel.setText("Loaded Blinking Dataset: exported_data.csv")
        
        # Enable the EM button since we now have data
        self.runEMButton.setEnabled(True)
        
        # Show a success message
        self.show_popup("Data Transferred", "Blink data has been successfully transferred to stoichiometry analysis.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
