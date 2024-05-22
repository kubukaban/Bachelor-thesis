import tkinter as tk
from tkinter import filedialog, messagebox, ttk # this is needed for dialogwindow
import re    #regex for text manipulation
import os    #librariry for uploading and saving files
import subprocess

# firstly we need to import neseceary libraires for this program
# this function bellow install library
def install(package):
    subprocess.check_call(["pip", "install", package])

# firstly we try import libraries, if they are not in the enviroment they are installed via function
try:
    import pandas as pd # pandas for data storage and manipulation
except ImportError:
    print("Pandas library is not installed. Installing...")
    install("pandas")
    import pandas as pd
    
try:
    import numpy as np # numpy for fast calculation
except ImportError:
    print("Numpy library is not installed. Installing...")
    install("numpy")
    import numpy as np

try:
    import matplotlib.pyplot as plt # library for vizualization
except ImportError:
    print("Matplotlib library is not installed. Installing...")
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    from tqdm import tqdm # library for progress bar
except ImportError:
    print("tqdm library is not installed. Installing...")
    install("tqdm")
    from tqdm import tqdm


# main body of the program is in the class Program
class Program:
    
    # initialization of the program, and definition of important atributtes
    def __init__(self):
        self.vertices1 = [] # list of vertices from 1. file
        self.faces1 = []    # list of faces from 1. file
        self.vertices2 = [] # list of vertices from 2. file
        self.faces2 = []    # list of faces from 2. file
        self.filename1 = '' # name of the first file
        self.filename2 = '' # name of the second file
        self.alpha1 = 0     # average angles of the files, firstly set to 0
        self.alpha2 = 0
        self.beta1 = 0
        self.beta2 = 0
        self.gamma1 = 0
        self.gamma2 = 0
        self.type = "boxplot" #type of plot, boxplot is default
        self.log_scale = False # Flag for logarithmic scale
        
        # Initialize Tkinter root window
        root = tk.Tk()
        root.title("Obj Plotter")
        
        # Load buttons for loading files
        load_button = tk.Button(root, text="Load first file", command=self.load_file1)
        load_button.pack()
        load_button = tk.Button(root, text="Load second file", command=self.load_file2)
        load_button.pack()
        
        # Threshold entry field, 2 is default
        self.threshold_entry = tk.Entry(width = 5)
        self.threshold_entry.insert(2, "2")
        self.threshold_entry.pack()
        
         # Buttons to choose plot type
        boxplot_button = tk.Button(root, text = "Boxplot", command = self.to_boxplot)
        boxplot_button.pack()
        
        scatterplot_button = tk.Button(root, text = "2D Scatterplot", command = self.to_scatter)
        scatterplot_button.pack()
        
        scatterplot3D_button = tk.Button(root, text = "3D Scatterplot", command = self.to_scatter3D)
        scatterplot3D_button.pack()
        
         # Radio buttons to choose measure
        self.radio_var = tk.IntVar()
        polomerov_button = tk.Radiobutton(root, text="Radius ratio", variable=self.radio_var, value=1, command=self.on_checked)
        polomerov_button.pack()
        extrem_stran_button = tk.Radiobutton(root, text="Edge ratio", variable=self.radio_var, value=2, command=self.on_checked)
        extrem_stran_button.pack()
        pomer_polomer_polobvod_button = tk.Radiobutton(root, text="Circumradius to half perimeter", variable=self.radio_var, value=3, command=self.on_checked)
        pomer_polomer_polobvod_button.pack()
        pomer_stran_button = tk.Radiobutton(root, text="Aspect ratio", variable=self.radio_var, value=4, command=self.on_checked)
        pomer_stran_button.pack()
        pomer_opisana_strana_button = tk.Radiobutton(root, text="Circumradius to edge", variable=self.radio_var, value=5, command=self.on_checked)
        pomer_opisana_strana_button.pack()
        priemer_button = tk.Radiobutton(root, text="Average triangles", variable=self.radio_var, value=6, command=self.on_checked)
        priemer_button.pack()
        
        # Checkbox for log scale
        log_var = tk.BooleanVar()
        log_checkbox = ttk.Checkbutton(root, text="Log scale", variable=log_var, command=lambda: self.check_scale(log_var))
        log_checkbox.pack()
        
        # Save button
        save_button = tk.Button(root, text = 'save', command = self.save)
        save_button.pack()
        
        # Start Tkinter main loop
        root.mainloop()
        
        
     # Function to save files   
    def save(self):
        print('save first file')
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Text files", "*.csv"), ("All files", "*.*")])
        if filepath:
            self.faces1.to_csv(path_or_buf = filepath)
            print("File saved at:", filepath)
        print('save second file')
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Text files", "*.csv"), ("All files", "*.*")])
        if filepath:
            self.faces2.to_csv(path_or_buf = filepath)
            print("File saved at:", filepath)

    # Function to set plot type to boxplot
    def to_boxplot(self):
        self.type = 'boxplot'
        self.on_checked()
     
    # Function to set plot type to 2D scatterplot
    def to_scatter(self):
        self.type = "scatter"
        self.on_checked()
    
    # Function to set plot type to 3D scatterplot
    def to_scatter3D(self):
        self.type = "3Dscatter"
        self.on_checked()
    
    # Function to check and apply log scale
    def check_scale(self,log_var):
        self.log_scale = log_var.get()
        self.on_checked()
    
    
    # Function to handle changes in plot and measure options
    def on_checked(self):
        options = {1:"Radius ratio", 2:"Edge ratio", 3: "Circumradius to half perimeter",
                   4:"Aspect ratio", 5:"Circumradius to edge", 6:"priemer"}
        selected = self.radio_var.get()
        if selected == 6:
            self.average()
        elif self.type == "boxplot":
            self.boxplot(options[selected])
        elif self.type == "scatter":
            self.scatterplot(options[selected])
        elif self.type == "3Dscatter":
            self.scatter3D(options[selected])
        
        
    # Function to load the first file    
    def load_file1(self):
        faces = []
        vertices = []
        file_path = filedialog.askopenfilename()
        if file_path:
            file_name = os.path.basename(file_path)
            self.filename1 = file_name
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.startswith('v '):
                          vertex = list(map(float, line.split()[1:]))
                          vertices.append(vertex)
                        elif line.startswith('f '):
                           points = line.split()[1:]
                           try:
                              triangle = [int(re.search(r'\d+(?=(//))',x).group()) for x in points]
                           except AttributeError:
                               try:
                                   triangle = [int(re.search(r'\d+(?=(/))',x).group()) for x in points]
                               except AttributeError:
                                   triangle = list(map(int,points))
                           faces.append(triangle)
                self.vertices1 = pd.DataFrame(vertices, columns = ["x","y","z"])
                self.faces1 = pd.DataFrame(faces, columns = ["A","B","C"])
                print(self.filename1)
                processed_data,avg_angles = self.compute_ratio(self.vertices1, self.faces1)
                self.alpha1, self.beta1, self.gamma1 = avg_angles
                return processed_data
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    # Function to load the second file
    def load_file2(self):
        faces = []
        vertices = []
        file_path = filedialog.askopenfilename()
        if file_path:
            file_name = os.path.basename(file_path)
            self.filename2 = file_name
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.startswith('v '):
                          vertex = list(map(float, line.split()[1:]))
                          vertices.append(vertex)
                        elif line.startswith('f '):
                           points = line.split()[1:]
                           try:
                              triangle = [int(re.search(r'\d+(?=(//))',x).group()) for x in points]
                           except AttributeError:
                               try:
                                   triangle = [int(re.search(r'\d+(?=(/))',x).group()) for x in points]
                               except AttributeError:
                                   triangle = list(map(int,points))
                           faces.append(triangle)
                self.vertices2 = pd.DataFrame(vertices, columns = ["x","y","z"])
                self.faces2 = pd.DataFrame(faces, columns = ["A","B","C"])
                print(self.filename2)
                processed_data,avg_angles = self.compute_ratio(self.vertices2, self.faces2)
                self.alpha2, self.beta2, self.gamma2 = avg_angles
                return processed_data
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    # Function to compute various ratios for vertices and faces
    def compute_ratio(self, vertices, faces):
        result = np.zeros((9, len(faces)))

        sin_alpha = np.zeros(len(faces))
        sin_beta = np.zeros(len(faces))
        sin_gamma = np.zeros(len(faces))
        dlhsie = np.zeros(len(faces))
        kratsie = np.zeros(len(faces))
        thety0 = np.zeros(len(faces))
        thetymax = np.zeros(len(faces))
        alphas = np.array([])
        betas = np.array([])
        gammas = np.array([])
        areas = np.zeros(len(faces))
        for i in tqdm(range(len(faces))):
            point_A = vertices.iloc[faces.iloc[i, 0] - 1]
            point_B = vertices.iloc[faces.iloc[i, 1] - 1]
            point_C = vertices.iloc[faces.iloc[i, 2] - 1]

            c = np.sqrt(np.sum((point_A - point_B) ** 2))
            a = np.sqrt(np.sum((point_B - point_C) ** 2))
            b = np.sqrt(np.sum((point_C - point_A) ** 2))

            alpha = np.arccos((-a ** 2 + b ** 2 + c ** 2) / (2 * c * b))
            beta = np.arccos((a ** 2 - b ** 2 + c ** 2) / (2 * a * c))
            gamma = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
            alphas= np.append(alphas, alpha)
            betas = np.append(betas, beta)
            gammas = np.append(gammas, gamma)
            dlhsie[i] = max(a, b, c)
            kratsie[i] = min(a, b, c)

            thety0[i] = min(alpha, beta, gamma)
            thetymax[i] = max(alpha, beta, gamma)

            sin_alpha[i] = np.sin(alpha)
            sin_beta[i] = np.sin(beta)
            sin_gamma[i] = np.sin(gamma)
            
            s = (a+b+c)/2
            # nejdem davat 0 sem
            areas[i] = np.sqrt(s*(s-a)*(s-b)*(s-c))
            result[:, i] = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Initialize
        areas = areas/np.min(areas[areas > 0]) #normalizacia ploch
        result[0, :] = (sin_alpha + sin_beta + sin_gamma) / (2 * sin_alpha * sin_beta * sin_gamma) # ro
        result[1, :] = dlhsie / kratsie # tau
        result[2, :] = 1 / (sin_alpha + sin_beta + sin_gamma) # nu
        result[3, :] = (np.sin(thety0) + np.sin(thetymax) + np.sin(thety0 + thetymax)) / (
                    np.sin(thety0) * np.sin(thety0 + thetymax)) # iota
        result[4, :] = 1 / (2 * np.sin(thetymax)) # omega
        result[5, :] = dlhsie
        result[6, :] = areas
        result[7, :] = thety0
        result[8, :] = thetymax
        faces['Radius ratio'] = result[0, :] / 2
        faces['Edge ratio'] = result[1, :]
        faces["Circumradius to half perimeter"] = result[2, :] * 3 * np.sqrt(3) / 2
        faces["Aspect ratio"] = result[3, :] / (2 * np.sqrt(3))
        faces["Circumradius to edge"] = result[4, :] * 2
        faces['najdlhsia'] = result[5, :]
        faces['obsahy'] = result[6]
        faces['min_uhol'] = result[7, :] 
        faces['max_uhol'] = result[8, :] 
        print('koniec')
        mask1 = ~np.isnan(alphas)
        mask2 = ~np.isnan(betas)
        mask3 = ~np.isnan(gammas)
        alphas = alphas[mask1]
        betas = betas[mask2]
        gammas = gammas[mask3]
        return faces,[np.mean(alphas),np.mean(betas),np.mean(gammas)]
    
    # Function to display average triangles 
    def average(self):
        plt.close()
        fig, ax = plt.subplots()
        theta0_1 = min(self.alpha1, self.beta1, self.gamma1)
        thetamax_1 = max(self.alpha1, self.beta1, self.gamma1)
        thetamid_1 = np.median((self.alpha1, self.beta1, self.gamma1))
        area1 = np.mean(self.faces1['obsahy'])
        area2 = np.mean(self.faces2['obsahy'])
        
        shortest1 = area1/((1/2)*np.sin(theta0_1))
        #law of sine
        middle1 = (shortest1 * np.sin(thetamid_1))/np.sin(theta0_1)
        longest1 = (shortest1 * np.sin(thetamax_1))/np.sin(thetamax_1)
        
        #draw firt triangle
        x0, y0 = 0, 0
        x1, y1 = shortest1, 0
        x2 = middle1 * np.cos(np.pi - thetamax_1) + shortest1
        y2 = middle1 * np.sin(thetamax_1) 
        plt.fill([x0,x1,x2],[y0,y1,y2], c='C0')
        
        theta0_2 = min(self.alpha2, self.beta2, self.gamma2)
        thetamax_2 = max(self.alpha2, self.beta2, self.gamma2)
        thetamid_2 = np.median((self.alpha2, self.beta2, self.gamma2))
        shortest2 = area2/((1/2)*np.sin(theta0_2))
        #law of sine
        middle2 = (shortest2 * np.sin(thetamid_2))/np.sin(theta0_2)
        longest2 = (shortest2 * np.sin(thetamax_2))/np.sin(thetamax_2)
        # draw second triangle
        x0, y0 = shortest1, 0
        x1, y1 = shortest1 + shortest2, 0
        x2 = middle2 * np.cos(np.pi - thetamax_2) + shortest2 + shortest1
        y2 = middle2 * np.sin(thetamax_2) 
        plt.fill([x0,x1,x2],[y0,y1,y2], c='C1')
        plt.xticks([1,shortest1], [self.filename1, self.filename2])
        plt.title('Priemerné trojuholníky oboch objektov')
        fig.text(0.5, 0.01, f'Priemerna plocha prveho='+str(np.mean(self.faces1['obsahy'])) + f'\nPrimerna plocha druheho='+str(np.mean(self.faces2['obsahy'])), ha='center', fontsize=10)
        plt.show()
        
    # Function to display boxplot for selected measure
    def boxplot(self,selected):
        plt.close()
        fig, ax = plt.subplots() 
        data1 = self.faces1[selected].dropna()
        data2 = self.faces2[selected].dropna()
       
        ax.boxplot(data1.dropna(), showfliers=False, positions=[1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='skyblue'))
        ax.boxplot(data2.dropna(), showfliers=False, positions=[2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        ax.set_xticks([1, 2])
        ax.set_xticklabels([self.filename1, self.filename2])
       
        text1 = f"Summary statistic\n{self.filename1} Min = {np.min(data1)}, 1.Qu = {np.quantile(data1,0.25)}, median = {np.median(data1)}, mean = {np.mean(data1)}, 3.Qu = {np.quantile(data1, 0.75)}, Max = {np.max(data1)}"
        text2 = f"\n{self.filename2} Min = {np.min(data2)}, 1.Qu = {np.quantile(data2,0.25)}, median = {np.median(data2)}, mean = {np.mean(data2)}, 3.Qu = {np.quantile(data2, 0.75)}, Max = {np.max(data2)}"
        fig.text(0.5, 0.01, text1 + text2, ha='center', fontsize=10)
            
        
        plt.show()
        
        
    # Function to display 2D scatterplot for selected measure
    def scatterplot(self,selected):
        data1 = self.faces1[selected]
        data2 = self.faces2[selected]
        thresh = float(self.threshold_entry.get())
        plt.close()
     
        fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,sharex=False, sharey=False)
        if self.log_scale:
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            ax3.set_yscale('log')
            ax4.set_yscale('log')
        # graph oh 1st file
        ax1.scatter(self.faces1['najdlhsia'].dropna()[data1 > thresh], data1[data1 > thresh], color='red', label='zle', s = 5)
        ax1.scatter(self.faces1['najdlhsia'].dropna()[data1 <= thresh], data1[data1 <= thresh], color='green', label='dobre', s = 5)
        ax1.set(xlabel = 'dlzka max hrany', ylabel = 'miera kvality', title = self.filename1)
        
        ax2.scatter(self.faces1['obsahy'].dropna()[data1 > thresh], data1[data1 > thresh], color='red', label='zle', s = 5)
        ax2.scatter(self.faces1['obsahy'].dropna()[data1 <= thresh], data1[data1 <= thresh], color='green', label='dobre', s = 5)
        ax2.set(xlabel='obsah', ylabel='miera kvality', title = self.filename1)
        
        fig.suptitle(selected)
        
        # graph of 2nd file
        ax3.scatter(self.faces2['najdlhsia'].dropna()[data2 > thresh], data2[data2 > thresh], color='red', label='zle', s = 5)
        ax3.scatter(self.faces2['najdlhsia'].dropna()[data2 <= thresh], data2[data2 <= thresh], color='green', label='dobre', s = 5)
        ax3.set(xlabel = 'dlzka max hrany', ylabel = 'miera kvality', title = self.filename2)
        
        ax4.scatter(self.faces2['obsahy'].dropna()[data2 > thresh], data2[data2 > thresh], color='red', label='zle', s = 5)
        ax4.scatter(self.faces2['obsahy'].dropna()[data2 <= thresh], data2[data2 <= thresh], color='green', label='dobre', s = 5)
        ax4.set(xlabel='obsah', ylabel='miera kvality', title = self.filename2)
        
        fig.suptitle(selected)
        
        plt.tight_layout()
        plt.show()
      
    # Function to display 3D scatterplot for selected measure  
    def scatter3D(self,selected):
        data1 = self.faces1[selected]
        data2 = self.faces2[selected]
        thresh = float(self.threshold_entry.get())
        plt.close()
        fig = plt.figure()
        # graphs of 1st file
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.set_xlabel('Najdlhsia strana')
        ax1.set_ylabel('obshah trojuholnika')
        ax1.set_zlabel('miera kvality')
        ax1.set_title(f'3D graf velkosti pre {self.filename1}')
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.set_xlabel('najmensi uhol')
        ax2.set_ylabel('najvacsi uhol')
        ax2.set_zlabel('miera kvality')
        ax2.set_title(f'3D uhlov pre {self.filename1}')
        # graphs of 2nd. file
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.set_xlabel('Najdlhsia strana')
        ax3.set_ylabel('obshah trojuholnika')
        ax3.set_zlabel('miera kvality')
        ax3.set_title(f'3D graf velkosti pre {self.filename2}')
        
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.set_xlabel('najmensi uhol')
        ax4.set_ylabel('najvacsi uhol')
        ax4.set_zlabel('miera kvality')
        ax4.set_title(f'3D uhlov pre {self.filename2}')
        
        if self.log_scale:# user choose log scale
            ax1.scatter(self.faces1['najdlhsia'].dropna()[data1 > thresh],self.faces1['obsahy'][data1 > thresh].dropna(),np.log(data1[data1 > thresh]), color = 'red', label = 'zle')
            ax1.scatter(self.faces1['najdlhsia'].dropna()[data1 <= thresh],self.faces1['obsahy'][data1 <= thresh].dropna(),np.log(data1[data1 <= thresh]), color = 'green', label = 'dobre')
            ax1.set_zscale('log')
            ax2.scatter(self.faces1['min_uhol'].dropna()[data1 > thresh],self.faces1['max_uhol'].dropna()[data1 > thresh],np.log(data1[data1 > thresh]), color = 'red', label = 'zle')
            ax2.scatter(self.faces1['min_uhol'].dropna()[data1 <= thresh],self.faces1['max_uhol'].dropna()[data1 <= thresh],np.log(data1[data1 <= thresh]), color = 'green', label = 'dobre')
            ax2.set_zscale('log')
            
            ax3.scatter(self.faces2['najdlhsia'].dropna()[data2 > thresh],self.faces2['obsahy'][data2 > thresh].dropna(),np.log(data2[data2 > thresh]), color = 'red', label = 'zle')
            ax3.scatter(self.faces2['najdlhsia'].dropna()[data2 <= thresh],self.faces2['obsahy'][data2 <= thresh].dropna(),np.log(data2[data2 <= thresh]), color = 'green', label = 'dobre')
            ax3.set_zscale('log')
            ax4.scatter(self.faces2['min_uhol'].dropna()[data2 > thresh],self.faces2['max_uhol'].dropna()[data2 > thresh],np.log(data2[data2 > thresh]), color = 'red', label = 'zle')
            ax4.scatter(self.faces2['min_uhol'].dropna()[data2 <= thresh],self.faces2['max_uhol'].dropna()[data2 <= thresh],np.log(data2[data2 <= thresh]), color = 'green', label = 'dobre')
            ax4.set_zscale('log')
        else: # user didnt choose log scale
            ax1.scatter(self.faces1['najdlhsia'].dropna()[data1 > thresh],self.faces1['obsahy'][data1 > thresh].dropna(),data1[data1 > thresh], color = 'red', label = 'zle')
            ax1.scatter(self.faces1['najdlhsia'].dropna()[data1 <= thresh],self.faces1['obsahy'][data1 <= thresh].dropna(),data1[data1 <= thresh], color = 'green', label = 'dobre')
            
            ax2.scatter(self.faces1['min_uhol'].dropna()[data1 > thresh],self.faces1['max_uhol'].dropna()[data1 > thresh],data1[data1 > thresh], color = 'red', label = 'zle')
            ax2.scatter(self.faces1['min_uhol'].dropna()[data1 <= thresh],self.faces1['max_uhol'].dropna()[data1 <= thresh],data1[data1 <= thresh], color = 'green', label = 'dobre')
            ax3.scatter(self.faces2['najdlhsia'].dropna()[data2 > thresh],self.faces2['obsahy'][data2 > thresh].dropna(),data2[data2 > thresh], color = 'red', label = 'zle')
            ax3.scatter(self.faces2['najdlhsia'].dropna()[data2 <= thresh],self.faces2['obsahy'][data2 <= thresh].dropna(),data2[data2 <= thresh], color = 'green', label = 'dobre')
            
            ax4.scatter(self.faces2['min_uhol'].dropna()[data2 > thresh],self.faces2['max_uhol'].dropna()[data2 > thresh],data2[data2 > thresh], color = 'red', label = 'zle')
            ax4.scatter(self.faces2['min_uhol'].dropna()[data2 <= thresh],self.faces2['max_uhol'].dropna()[data2 <= thresh],data2[data2 <= thresh], color = 'green', label = 'dobre')
                                  
        ax1.legend()
        ax2.legend()
        plt.show()
        
        
if __name__ == "__main__":
    p = Program()