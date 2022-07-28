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
from tkinter import ttk


class App(tk.Tk):
    def __init__(self): 
        # Instanciate
        super().__init__()
        self.title('Spectrum Labelling App')

        # Specify Output file (will get created if not existent)
        self.file_path = os.getcwd() + '\\LabeledSpectra.txt'
        self.file_name = ""

        # Initialize directory
        self.top_dir = os.getcwd()

        # Create App
        self.createWidgets()


    def createWidgets(self):
        """Create App main functionalities."""
        self.rowconfigure(0, minsize=400, weight=1)
        self.columnconfigure(1, minsize=400, weight=1)
        padding = {'padx': 5, 'pady': 5}

        # Create top layer frames
        self.FrameFig = tk.Frame(master=self)
        self.FrameFig.grid(row=0, column=0)
        self.FrameLabel = tk.Frame(master=self)
        self.FrameLabel.grid(row=0, column=1)

        # create FigureCanvasTkAgg object
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.FrameFig)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        NavigationToolbar2Tk(self.canvas, self.FrameFig)

        # Add user entry
        userFrame = tk.Frame(master=self.FrameLabel)
        tk.Label(master=userFrame, text="Enter user name: ").pack(side=tk.LEFT, **padding)
        self.user_entry = tk.Entry(master=userFrame)
        self.user_entry.insert(0, "Username")
        self.user_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, **padding)

        # Create Radio Buttons
        radioBtnFrame = tk.Frame(master=self.FrameLabel)
        signal = tk.StringVar(name="Signal", value="0")
        noise = tk.StringVar(name="Noise", value="0")
        distinctness = tk.StringVar(name="Distinctness of Peaks", value="0")
        backNoise = tk.StringVar(name="Background Noise", value="0")
        peakWidth = tk.StringVar(name="Peak Width", value="0")
        categories = [signal, noise, distinctness, backNoise, peakWidth]
        labels = ["0", "1", "2", "3", "4"]
        for idx, category in enumerate(categories):
            tk.Label(radioBtnFrame, text=category).grid(row=idx,column=0, **padding)
            for jdx, label in enumerate(labels):
                ttk.Radiobutton(radioBtnFrame, text=label, variable=category, value=int(label)).grid(row=idx,column=jdx+1, **padding)
        self.labels = categories

        # Create Buttons
        btnFrame = tk.Frame(master=self.FrameLabel)
        submit_btn = tk.Button(master=btnFrame, text="Submit", command=self.submit)
        submit_btn.pack(side=tk.RIGHT, **padding)

        skip_btn = tk.Button(master=btnFrame, text="Skip", command=self.startFromTopDir)
        skip_btn.pack(side=tk.RIGHT, **padding)

        userFrame.grid(row=0, column=0, padx=5, pady=10)
        radioBtnFrame.grid(row=1, column=0, padx=5, pady=10)
        btnFrame.grid(row=2, column=0, padx=5, pady=10)

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
        menubar.add_cascade(label="Menu", menu=open_menu)
        menubar.add_cascade(label="Label File", menu=label_menu)
 

    def open_folder(self):
        """Open a folder with relevant subfolders/files."""
        self.top_dir = askdirectory()
        self.startFromTopDir()


    def startFromTopDir(self):
        os.chdir(self.top_dir)
        self.open_file()


    def open_file(self):
        """Recursive method that browses for a random file in random subfolders."""
        self.folders = [folder for folder in os.listdir(".") if os.path.isdir(folder)]
        if self.folders:
            n = len(self.folders)
            k = rnd.randint(0, n-1)
            random_folder = self.folders[k]
            os.chdir(os.getcwd() + "\\" + random_folder)
            self.open_file()
        else:
            files = glob.glob(os.getcwd() + '/*.' + 'DAT')
            if files:
                n = len(files)
                k = rnd.randint(0, n-1)
                random_file = files[k]
                with open(random_file) as f:
                    # remove file ending and extract infos from file name for spatial coordinates
                    self.file_name = "".join(os.path.basename(random_file).split('.')[:-1])
                    # read wavelengths
                    lines = f.readlines()
                    w_raw = [line.split()[0] for line in lines]
                    w = np.asarray(w_raw).astype(float)
                    # read spectrum
                    spectrum_raw = [line.split()[1] for line in lines]
                    spectrum = np.asarray(spectrum_raw).astype(float)
                self.plot_spectrum(w, spectrum)


    def plot_spectrum(self, wavelengths, amplitudes):
        """Update figure and canvas."""
        self.figure.clear()
        axes = self.figure.add_subplot()
        axes.plot(wavelengths, amplitudes)
        axes.set_title('Spectrum')
        axes.set_ylabel('Intensity')
        axes.set_xlabel('Wavelength')
        self.canvas.draw()


    def submit(self):
        """Save labeled text."""
        with open(self.file_path, mode="a", encoding="utf-8") as output_file:
            # write file name
            output_file.write("file " + os.path.basename(os.getcwd()) + '\\' + self.file_name)
            # write labels
            output_file.write(" label ")
            for label in self.labels:
                output_file.write(label.get() + " ")
                label.set("0")
            # write user
            user = self.user_entry.get()
            output_file.write(" user " + user)
            # write date
            output_file.write(" date " + datetime.today().strftime('%Y-%m-%d') + "\n")
        self.startFromTopDir()


    def undo(self):
        with open(self.file_path, "r+") as f:
            lines = f.readlines()
            if lines:
                f.seek(0)
                for line in lines:
                    if line != lines[-1]:
                        f.write(line)
                f.truncate()


    def openTxtWindow(self):
        window = tk.Tk()
        sy = tk.Scrollbar(window)
        sx = tk.Scrollbar(window,  orient=tk.HORIZONTAL)
        editor = tk.Text(window, height=500, width=300, wrap='none')
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        editor.pack(side=tk.LEFT, fill=tk.Y)
        sy.config(command=editor.yview)
        sx.config(command=editor.xview)
        with open(self.file_path, mode="r", encoding="utf-8") as file:
            txt = file.read()
            editor.insert(tk.END, txt)
        window.mainloop()


if __name__ == '__main__':
    app = App()
    app.mainloop()
