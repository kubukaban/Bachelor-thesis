### Docunetation for the program

#Program Documentation
Overview
This program, encapsulated in the Program class, is designed to visualize and analyze 3D objects stored in files. It allows users to load data from two different files, process the data to compute various geometric properties, and display the results using different types of plots. The user interface is built using tkinter.

Class: Program
Attributes
vertices1: List of vertices from the first file.
faces1: List of faces from the first file.
vertices2: List of vertices from the second file.
faces2: List of faces from the second file.
filename1: Name of the first file.
filename2: Name of the second file.
alpha1, beta1, gamma1: Average angles for the first file.
alpha2, beta2, gamma2: Average angles for the second file.
type: Type of plot to be displayed ("boxplot" by default).
log_scale: Boolean indicating whether to use a logarithmic scale for plots.
radio_var: tk.IntVar to store the value of the selected radio button.
threshold_entry: tk.Entry widget for the threshold value.
Methods
__init__(self)
Initializes the program, sets up the GUI, and defines important attributes.

save(self)
Saves the faces data of both files to CSV files.

to_boxplot(self)
Sets the plot type to "boxplot" and updates the display.

to_scatter(self)
Sets the plot type to "scatter" and updates the display.

to_scatter3D(self)
Sets the plot type to "3Dscatter" and updates the display.

check_scale(self, log_var)
Updates the log_scale attribute based on the state of the log scale checkbox and refreshes the display.

on_checked(self)
Updates the display based on the selected measure and plot type.

load_file1(self)
Loads the first file, processes its data, and updates the vertices and faces attributes.

load_file2(self)
Loads the second file, processes its data, and updates the vertices and faces attributes.

compute_ratio(self, vertices, faces)
Computes various geometric ratios for the provided vertices and faces and returns the processed data along with average angles.

average(self)
Displays a plot comparing the average triangles of both objects.

boxplot(self, selected)
Displays a boxplot of the selected measure for both files.

scatterplot(self, selected)
Displays 2D scatter plots of the selected measure for both files.

scatter3D(self, selected)
Displays 3D scatter plots of the selected measure for both files.

Usage
Loading Files
Use the "Load first file" and "Load second file" buttons to load the data files. The files should be in a format where vertices are defined with lines starting with 'v ' and faces with lines starting with 'f '.
Choosing Plot Type
Select the plot type using the "Boxplot", "2D Scatterplot", and "3D Scatterplot" buttons.
Choosing Measure
Select the measure to be analyzed using the radio buttons. Options include various geometric ratios like pomer_polomerov, pomer_extremnych stran, etc.
Threshold and Log Scale
Set the threshold value in the threshold entry box.
Use the "Log scale" checkbox to toggle logarithmic scaling for the plots.
Saving Data
Use the "save" button to save the processed faces data to CSV files.
Dependencies
tkinter: For the graphical user interface.
pandas: For data manipulation and storage.
numpy: For numerical calculations.
matplotlib: For plotting graphs.
tqdm: For displaying progress bars.
os: For file path operations.
re: For regular expressions in parsing face data.
