import tkinter as tk
import os
import glob
import random as rnd
import numpy as np

import matplotlib

matplotlib.use('TkAgg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

from tkinter.filedialog import askopenfilename, asksaveasfilename


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Tkinter Matplotlib Demo')

        self.file_dir = os.getcwd() + '\\..\\Spectra Analysis\\sample'
        os.chdir(self.file_dir)

        # create a figure
        self.figure = Figure(figsize=(6, 4), dpi=100)

        # create FigureCanvasTkAgg object
        figure_canvas = FigureCanvasTkAgg(self.figure, self)
        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # create the toolbar
        NavigationToolbar2Tk(figure_canvas, self)

        file_name, w, spectrum = self.open_file(self.file_dir)

        label = tk.Label(master=self, text=file_name)
        label.pack()

        
    def open_file(self, dir):
        """Open a file for labeling."""
        files = glob.glob(dir + '/*.' + 'DAT')
        n = len(files)
        k = rnd.randint(0, n-1)
        print("Selecting file {0} out of {1}".format(k,n))
        random_file = files[k]
        with open(random_file) as f:
            # remove file ending and extract infos from file name for spatial coordinates
            file_name = os.path.basename(random_file).split('.')[:-1]
            # read spectrum
            lines = f.readlines()
            spectrum_raw = [line.split()[1] for line in lines]
            spectrum = np.asarray(spectrum_raw).astype(float)
            # read wavelengths
            w_raw = [line.split()[0] for line in lines]
            w = np.asarray(w_raw).astype(float)

        self.figure.clear()
        axes = self.figure.add_subplot()

        # create the barchart
        axes.plot(w, spectrum)
        axes.set_title('Spectrum')
        axes.set_ylabel('Intensity')
        axes.set_xlabel('Wavelength')
        return file_name, w, spectrum

    #def plot_spectrum(self, wavelengths, amplitudes):



if __name__ == '__main__':

    app = App()
    app.mainloop()