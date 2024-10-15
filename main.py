import sys, os
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox
from PyQt6 import uic
from PyQt6.QtCore import Qt
from MixtureModelAlgorithm import EM1, EM2, EM3  # Import from the original script
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import math


print("Initializing UI")

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        ui_path = self.resource_path("main.ui")

        uic.loadUi(ui_path, self)  # Load the UI file

        # Connect the buttons to their respective functions
        self.runButton.clicked.connect(self.run_replicates)
        # Connect Menu items to their functions
        self.actionLoadFile.triggered.connect(self.load_file)
        self.actionGraph_Dataset.triggered.connect(self.plot_dataset)
        self.radioEM1.clicked.connect(self.set_default_pi)
        self.radioEM2.clicked.connect(self.set_default_pi)
        self.radioEM3.clicked.connect(self.set_default_pi)


        self.inputLambda.setText("0")
        self.inputTheta.setText("1")

        # self.paramWidget.setStyleSheet("border: 1px solid red")

        self.data = None  # Placeholder for the loaded data
        self.data_imported = False
        self.lab_ineff = False

        self.replicates = 1
        self.subset_factor = 1

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

        # Plot some data
        # self.ax.plot([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])

    def resource_path(self, relative_path):
        """ Get the absolute path to the resource, works in development and after PyInstaller packaging """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def load_file(self):
        # Open a file dialog to select the CSV file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Data File", "", "CSV Files (*.csv)")
        
        if file_path:
            # Display the file path in the UI
            self.filePathLabel.setText(f"Loaded file: {file_path}")

            # Load the CSV file using numpy
            self.data = np.genfromtxt(file_path, delimiter=",")
            self.data_imported = True
        elif self.data_imported == True:
            pass
        else:
            self.filePathLabel.setText("No file loaded")

    def set_default_pi(self):
        if self.radioEM1.isChecked():
            self.inputPi.setText("0")
        elif self.radioEM2.isChecked():
            self.inputPi.setText("0,0")
        else:
            self.inputPi.setText("0,0,0")

    def run_em_algorithm(self):
        # Ensure data is loaded
        if self.data is None:
            # self.resultDisplay.setText("Please load a data file before running the algorithm")
            self.show_popup("Missing data", "Please load a data file before running the algorithm")
            return

        # Get user input for pi and lambda
        pi_input = self.inputPi.text()
        lambda_input = self.inputLambda.text()
        theta_input = self.inputTheta.text()

        # Process the input (convert string to list and float)
        try:
            pi = [float(x) for x in pi_input.split(',')] if pi_input else None
        except ValueError:
            self.show_popup("Invalid input", "Please enter the values in a comma-separated format")
            return

        lam = float(lambda_input) if lambda_input else None
        theta = float(theta_input) if theta_input else None

        self.lab_ineff = True if theta else False
        
        # Select EMX based on user selection
        if self.radioEM1.isChecked():
            self.run_em1(pi, lam)
        elif self.radioEM2.isChecked():
            self.run_em2(pi, lam, theta)
        elif self.radioEM3.isChecked():
            self.run_em3(pi, lam, theta)
        else:
            # self.resultDisplay.setText("Please select an algorithm")
            self.show_popup("Missing algorithm", "Please select an algorithm")
    
    def run_replicates(self):
        # Ensure data is loaded
        if self.data is None:
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

        self.display_results(lam_means, pi_means, aic_means, model)
        self.plot_result_with_error(pi_means, pi_stds, model)


    def run_em1(self, pi=[0], lam=0):
        Blinks = self.data

        # Initialize and run EM1
        em1 = EM1(Blinks, pi=pi, lam=lam)
        em1.initialize()
        em1.run()

        # Display the results in the QTextEdit widget
        self.display_results(em1.lam, em1.pi, em1.AIC, "M")
        self.plot_result(em1.pi, "M")


    def run_em2(self, pi=[0,0], lam=0, theta=1):
        Blinks = self.data

        # Initialize and run EM2
        em2 = EM2(Blinks, pi=pi, lam=lam)
        em2.initialize()
        em2.run()

        if self.lab_ineff:
            em2.theta = theta
            em2.apply_lab_ineff()

        # Display the results in the QTextEdit widget
        self.display_results(em2.lam, em2.pi, em2.AIC, "M/D")
        self.plot_result(em2.pi, "M/D")


    def run_em3(self, pi=[0,0,0], lam=0, theta=1):
        Blinks = self.data

        # Initialize and run EM3
        em3 = EM3(Blinks, pi=pi, lam=lam)
        em3.initialize()
        em3.run()

        if self.lab_ineff:
            em3.theta = theta
            em3.apply_lab_ineff()

        # Display the results in the QTextEdit widget
        self.display_results(em3.lam, em3.pi, em3.AIC, "M/D/T")
        self.plot_result(em3.pi, "M/D/T")

    def bunch_dye_simple(self, x, pi1, pi2, pi3, size):
        """Generate synthetic data"""
        x1 = np.random.choice(x, int(pi1 * size), replace=True)
        x2 = np.random.choice(x, int(pi2 * size), replace=True) + np.random.choice(x, int(pi2 * size), replace=True)
        x3 = np.random.choice(x, int(pi3 * size), replace=True) + np.random.choice(x, int(pi3 * size), replace=True) + np.random.choice(x, int(pi3 * size), replace=True)
        return np.concatenate((x1, x2, x3), axis=None)

    def display_results(self, lam, pi, AIC, model):
        # Display the results in the QTextEdit widget
        result_text = f"Estimated lambda: {lam}\nEstimated pi: {pi}\nAIC: {AIC}"
        # self.resultDisplay.setText(result_text)
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

    def plot_result(self, values, model):
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
        self.canvas.draw()

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

    def plot_dataset(self):
        
        if not self.data_imported:
            # self.resultDisplay.setText("Please import the data file before plotting")
            self.show_popup("Missing data", "Please import the data file before plotting")

            return

        self.ax.clear()

        # Plot the data
        self.ax.plot(sorted(self.data))
        # Add labels and a title
        self.ax.set_xlabel("Dye (Sorted)")
        self.ax.set_ylabel("Number of Blinks")

        self.canvas.draw()

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
        print(math.floor(len(self.data)*size_fraction))
        return [np.random.choice(self.data, size=math.floor(len(self.data)*size_fraction), replace=True) for _ in range(replicates)]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
