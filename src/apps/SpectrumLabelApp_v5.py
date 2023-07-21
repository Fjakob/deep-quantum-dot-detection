"""
Labelling App Version 5

Second release for quantum spectrum labelling
"""
import sys
sys.path.insert(0, '')

import tkinter as tk
import os
import glob
import random as rnd
import numpy as np
from datetime import datetime

import matplotlib
from matplotlib.figure import Figure
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from tkinter.filedialog import askdirectory

from src.apps.features.slider import Slider


class App(tk.Tk):
    def __init__(self): 
        # Instanciate
        super().__init__()
        self.title('Spectrum Labelling App')

        # Specify Output file (will get created if not existent)
        self.version = '_v5'
        self.file_path = None
        self.app_dir = os.getcwd()
        self.file_name = ""

        # Initialize directory
        self.top_dir = os.getcwd()

        # Create App
        self.createWidgets()


    def createWidgets(self):
        """Create App main functionalities."""
        self.rowconfigure(0, minsize=400, weight=1)
        self.columnconfigure(1, minsize=400, weight=1)
        padding = {'padx': 5, 'pady': 10}

        # Create top layer frames
        self.FrameFig = tk.Frame(master=self)
        self.FrameFig.grid(row=0, column=0, sticky="nsew")
        self.FrameLabel = tk.Frame(master=self)
        self.FrameLabel.grid(row=0, column=1)

        # create FigureCanvasTkAgg object
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.FrameFig)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        NavigationToolbar2Tk(self.canvas, self.FrameFig)

        # Add user entry
        userFrame = tk.Frame(master=self.FrameLabel)
        tk.Label(master=userFrame, text="Enter user name: ").pack(side=tk.LEFT, fill=tk.BOTH, **padding)
        self.user_entry = tk.Entry(master=userFrame)
        self.user_entry.insert(0, "Username")
        self.user_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, **padding)

        # Create Peak Counter
        peakCountFrame = tk.Frame(master=self.FrameLabel)

        # Peak Counter
        tk.Label(master=peakCountFrame, text="Number of Peaks").grid(row=0, column=0, **padding)
        self.lbl_value = tk.Label(master=peakCountFrame, text="0")
        self.lbl_value.grid(row=0, column=1, **padding)
        tk.Button(master=peakCountFrame, text="-", command=self.decrease).grid(row=0, column=2, sticky="nsew")
        tk.Button(master=peakCountFrame, text="+", command=self.increase).grid(row=0, column=3, sticky="nsew")

        # Create Slide Bar
        sliderBarFrame = tk.Frame(master=self.FrameLabel)
        self.slider = Slider(master=sliderBarFrame, min_val=0, max_val=100, init_lis = [50])
        self.slider.grid(row=0, column=0, **padding)

        # Create Buttons
        btnFrame = tk.Frame(master=self.FrameLabel)
        tk.Button(master=btnFrame, text="Submit", command=self.submit).pack(side=tk.RIGHT, fill=tk.BOTH, **padding)
        tk.Button(master=btnFrame, text="Skip", command=self.startFromTopDir).pack(side=tk.RIGHT, fill=tk.BOTH, **padding)

        # Add Frames to Label Frame
        userFrame.grid(row=0, column=0, **padding, sticky="nswe")
        peakCountFrame.grid(row=1, column=0, **padding, sticky="nswe")
        sliderBarFrame.grid(row=2, column=0, **padding, sticky="nswe")
        btnFrame.grid(row=3, column=0, **padding, sticky="nswe")

        # Create menu bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        open_menu = tk.Menu(menubar, tearoff=False)
        open_menu.add_command(label='Open Folder', command=self.open_folder)
        open_menu.add_separator()
        open_menu.add_command(label='Exit', command=self.destroy)
        label_menu = tk.Menu(menubar, tearoff=False)
        label_menu.add_command(label='Undo Last Submit', command=self.undo)
        label_menu.add_command(label='Open Label Text File', command=self.openTxtWindow)
        info_menu = tk.Menu(menubar, tearoff=False)
        info_menu.add_command(label="Info", command=self.openInfo)
        menubar.add_cascade(label="Menu", menu=open_menu)
        menubar.add_cascade(label="Label File", menu=label_menu)
        menubar.add_cascade(label="Info", menu=info_menu)


    def open_folder(self):
        """Asks for folder with relevant subfolders/files."""
        self.top_dir = askdirectory()
        if self.top_dir:
            self.startFromTopDir()


    def startFromTopDir(self):
        """Starts browsing from top directory."""
        os.chdir(self.top_dir)
        self.open_file()


    def open_file(self):
        """Recursive method that browses for a random file in a random subfolder."""
        folders = [folder for folder in os.listdir(".") if os.path.isdir(folder)]
        if folders:
            n = len(folders)
            k = rnd.randint(0, n-1)
            random_folder = folders[k]
            os.chdir(os.getcwd() + "\\" + random_folder)
            self.open_file()
        else:
            # search for '.DAT' files
            files = glob.glob(os.getcwd() + '/*.' + 'DAT')
            if files:
                n = len(files)
                k = rnd.randint(0, n-1)
                random_file = files[k]
                with open(random_file) as f:
                    # remove file ending
                    self.file_name = ".".join(os.path.basename(random_file).split('.')[:-1])
                    # read wavelengths
                    lines = f.readlines()
                    w_raw = [line.split()[0] for line in lines]
                    w = np.asarray(w_raw).astype(float)
                    self.current_w_range = int(np.floor(w[-1] - w[0]))
                    # read spectrum
                    spectrum_raw = [line.split()[1] for line in lines]
                    spectrum = np.asarray(spectrum_raw).astype(float)
                self.plot_spectrum(w, spectrum)


    def plot_spectrum(self, wavelengths, amplitudes):
        """Plots a spectrum."""
        self.figure.clear()
        axes = self.figure.add_subplot()
        axes.plot(wavelengths, amplitudes)
        axes.set_title('Spectrum')
        axes.set_ylabel('Intensity')
        axes.set_xlabel('Wavelength')
        axes.grid()
        self.canvas.draw()


    def increase(self):
        "Peak Counter Increment"
        value = int(self.lbl_value["text"])
        self.lbl_value["text"] = f"{value + 1}"

    def decrease(self):
        "Peak Counter Decrement"
        value = int(self.lbl_value["text"])
        if value != 0:
            self.lbl_value["text"] = f"{value - 1}"

    
    def createOutputFile(self):
        "Creates Output File."
        self.file_path = self.app_dir + '\\LabeledSpectra_' + self.user_entry.get() + self.version + '.txt'
        with open(self.file_path, mode="a", encoding="utf-8") as output_file:
            output_file.write('')


    def submit(self):
        """Save labeled spectrum in textfile."""
        if self.file_path is None:
            self.createOutputFile()
        with open(self.file_path, mode="a", encoding="utf-8") as output_file:
            # write labels
            output_file.write("label " + self.lbl_value["text"] + " ")
            self.lbl_value["text"] = f"{0}"
            output_file.write("{:.2f} ".format(np.mean(self.slider.getValues())/100))
            # write w_range
            output_file.write(f"w_range {self.current_w_range}")
            # write date
            output_file.write(" date " + datetime.today().strftime('%Y-%m-%d'))
            # write user
            user = self.user_entry.get()
            output_file.write(" user " + user)
            # write file name
            output_file.write(" file " + os.path.basename(os.getcwd()) + '\\' + self.file_name + "\n")
        self.startFromTopDir()


    def undo(self):
        """Remove last line from textfile."""
        with open(self.file_path, "r+") as f:
            lines = f.readlines()
            if lines:
                f.seek(0)
                for line in lines:
                    if line != lines[-1]:
                        f.write(line)
                f.truncate()


    def openTxtWindow(self):
        """Open textfile in external text viewer."""
        if self.file_path is None:
            self.createOutputFile() 
        with open(self.file_path, mode="r", encoding="utf-8") as file:
            txt = file.read()
        self.createTextViewer(title=os.path.basename(self.file_path), text=txt)


    def openInfo(self):
        """Open Info in external text viewer."""
        info = "Number of Peaks: Number of clearly visible peaks. Ignore very small peaks in high intensity spectra. \n"
        info += "\nSpectrum Impression: Overall subjective impression of the spectrum. Would you like to work with this one? ++ means perfect. \n"
        info += "\nBackground: Signal not matching the noise or peaks. Bulbs, gradient, etc. ++ means no background. \n"
        info += "\nDistinctiveness of Peaks: How well are the peaks isolated (e.g. douple peaks). Prioritize bright peaks over dark ones. ++ means perfecr. \n"
        info += "\nPeak Width: Width of the individual peaks (e.g shoulders) without broadening by neighbours (e.g. double peaks). ++ means narrow. \n"
        self.createTextViewer(title='Info', text=info)


    def createTextViewer(self, title, text):
        """Creates external text viewer"""
        window = tk.Tk()
        window.title(title)
        # create text field and scrollbars
        editor = tk.Text(window, height=30, width=100, wrap='none')
        sy = tk.Scrollbar(window)
        sx = tk.Scrollbar(window,  orient=tk.HORIZONTAL)
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        editor.pack(side=tk.LEFT, fill='both', expand=True)
        # link scrollbars to text field
        sy.config(command=editor.yview)
        sx.config(command=editor.xview)
        # insert text
        editor.insert(tk.END, text)
        window.mainloop()


if __name__ == '__main__':
    app = App()
    app.mainloop()
