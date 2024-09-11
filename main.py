import sys, os
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem
from PyQt6 import uic
from PyQt6.QtCore import Qt
from MixtureModelAlgorithm import EM1, EM2, EM3  # Import from the original script

print("Initializing UI")

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        ui_path = self.resource_path("main.ui")

        uic.loadUi(ui_path, self)  # Load the UI file

        # Connect the buttons to their respective functions
        self.runButton.clicked.connect(self.run_em_algorithm)
        self.loadFileButton.clicked.connect(self.load_file)

        # Connect Menu items to their functions
        self.actionLoadFile.triggered.connect(self.load_file)

        self.data = None  # Placeholder for the loaded data
        self.data_imported = False

    def resource_path(self, relative_path):
        """ Get the absolute path to the resource, works in development and after PyInstaller packaging """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def add_result_entry(self, parameter, value):
        # Create two items for each row
        item1 = QStandardItem(parameter)
        item2 = QStandardItem(value)

        # Add the items to the model as a new row
        self.model.appendRow([item1, item2])

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

    def run_em_algorithm(self):
        # Ensure data is loaded
        if self.data is None:
            self.resultDisplay.setText("Please load a data file before running the algorithm.")
            return

        # Get user input for pi and lambda
        pi_input = self.inputPi.text()
        lambda_input = self.inputLambda.text()
        
        # Process the input (convert string to list and float)
        pi = [float(x) for x in pi_input.split(',')] if pi_input else None
        lam = float(lambda_input) if lambda_input else None
        
        # Select EM2 or EM3 based on user selection
        if self.radioEM1.isChecked():
            self.run_em1(pi, lam)
        elif self.radioEM2.isChecked():
            self.run_em2(pi, lam)
        elif self.radioEM3.isChecked():
            self.run_em3(pi, lam)
        else:
            self.resultDisplay.setText("Please select EM2 or EM3.")

    def run_em1(self, pi=[0], lam=0):
        Blinks = self.data

        # Initialize and run EM1
        em1 = EM1(Blinks, pi=pi, lam=lam)
        em1.initialize()
        em1.run()

        # Display the results in the QTextEdit widget
        self.display_results(em1.lam, em1.pi, em1.AIC, "M")

    def run_em2(self, pi=[0,0], lam=0):
        Blinks = self.data

        # Initialize and run EM2
        em2 = EM2(Blinks, pi=pi, lam=lam)
        em2.initialize()
        em2.run()

        # Display the results in the QTextEdit widget
        self.display_results(em2.lam, em2.pi, em2.AIC, "M/D")

    def run_em3(self, pi=[0,0,0], lam=0):
        Blinks = self.data

        # Initialize and run EM3
        em3 = EM3(Blinks, pi=pi, lam=lam)
        em3.initialize()
        em3.run()

        # Display the results in the QTextEdit widget
        self.display_results(em3.lam, em3.pi, em3.AIC, "M/D/T")

    def bunch_dye_simple(self, x, pi1, pi2, pi3, size):
        """Generate synthetic data"""
        x1 = np.random.choice(x, int(pi1 * size), replace=True)
        x2 = np.random.choice(x, int(pi2 * size), replace=True) + np.random.choice(x, int(pi2 * size), replace=True)
        x3 = np.random.choice(x, int(pi3 * size), replace=True) + np.random.choice(x, int(pi3 * size), replace=True) + np.random.choice(x, int(pi3 * size), replace=True)
        return np.concatenate((x1, x2, x3), axis=None)

    def display_results(self, lam, pi, AIC, model):
        # Display the results in the QTextEdit widget
        result_text = f"Estimated lambda: {lam}\nEstimated pi: {pi}\nAIC: {AIC}"
        self.resultDisplay.setText(result_text)
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

        item = QTableWidgetItem(model)
        self.tableWidget.setItem(row_position, 4, item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
