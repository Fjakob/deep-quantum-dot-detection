"""
Labelling App Version 5

Second release for quantum spectrum labelling
"""
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

from tkinter import *
from tkinter.ttk import *


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


class Slider(Frame):
    LINE_COLOR = "#476b6b"
    LINE_WIDTH = 3
    BAR_COLOR_INNER = "#5c8a8a"
    BAR_COLOR_OUTTER = "#c2d6d6"
    BAR_RADIUS = 10
    BAR_RADIUS_INNER = BAR_RADIUS - 5
    DIGIT_PRECISION = ".1f"  # for showing in the canvas

    def __init__(
        self,
        master,
        width=400,
        height=80,
        min_val=0,
        max_val=1,
        init_lis=None,
        show_value=True,
        removable=False,
        addable=False,
    ):
        Frame.__init__(self, master, height=height, width=width)
        self.master = master
        if init_lis == None:
            init_lis = [min_val]
        self.init_lis = init_lis
        self.max_val = max_val
        self.min_val = min_val
        self.show_value = show_value
        self.H = height
        self.W = width
        self.canv_H = self.H
        self.canv_W = self.W
        if not show_value:
            self.slider_y = self.canv_H / 2  # y pos of the slider
        else:
            self.slider_y = self.canv_H * 2 / 5
        self.slider_x = Slider.BAR_RADIUS  # x pos of the slider (left side)

        self.bars = []
        self.selected_idx = None  # current selection bar index
        for value in self.init_lis:
            pos = (value - min_val) / (max_val - min_val)
            ids = []
            bar = {"Pos": pos, "Ids": ids, "Value": value}
            self.bars.append(bar)

        self.canv = Canvas(self, height=self.canv_H, width=self.canv_W)
        self.canv.pack()
        self.canv.bind("<Motion>", self._mouseMotion)
        self.canv.bind("<B1-Motion>", self._moveBar)
        if removable:
            self.canv.bind("<3>", self._removeBar)
        if addable:
            self.canv.bind("<ButtonRelease-1>", self._addBar)

        self.__addTrack(
            self.slider_x, self.slider_y, self.canv_W - self.slider_x, self.slider_y
        )
        for bar in self.bars:
            bar["Ids"] = self.__addBar(bar["Pos"])

    def getValues(self):
        values = [bar["Value"] for bar in self.bars]
        return sorted(values)

    def _mouseMotion(self, event):
        x = event.x
        y = event.y
        selection = self.__checkSelection(x, y)
        if selection[0]:
            self.canv.config(cursor="hand2")
            self.selected_idx = selection[1]
        else:
            self.canv.config(cursor="")
            self.selected_idx = None

    def _moveBar(self, event):
        x = event.x
        y = event.y
        if self.selected_idx == None:
            return False
        pos = self.__calcPos(x)
        idx = self.selected_idx
        self.__moveBar(idx, pos)

    def _removeBar(self, event):
        x = event.x
        y = event.y
        if self.selected_idx == None:
            return False
        idx = self.selected_idx
        ids = self.bars[idx]["Ids"]
        for id in ids:
            self.canv.delete(id)
        self.bars.pop(idx)

    def _addBar(self, event):
        x = event.x
        y = event.y

        if self.selected_idx == None:
            pos = self.__calcPos(x)
            ids = []
            bar = {
                "Pos": pos,
                "Ids": ids,
                "Value": self.__calcPos(x) * (self.max_val - self.min_val)
                + self.min_val,
            }
            self.bars.append(bar)

            for i in self.bars:
                ids = i["Ids"]
                for id in ids:
                    self.canv.delete(id)

            for bar in self.bars:
                bar["Ids"] = self.__addBar(bar["Pos"])

    def __addTrack(self, startx, starty, endx, endy):
        id1 = self.canv.create_line(
            startx, starty, endx, endy, fill=Slider.LINE_COLOR, width=Slider.LINE_WIDTH
        )
        return id

    def __addBar(self, pos):
        """@ pos: position of the bar, ranged from (0,1)"""
        if pos < 0 or pos > 1:
            raise Exception("Pos error - Pos: " + str(pos))
        R = Slider.BAR_RADIUS
        r = Slider.BAR_RADIUS_INNER
        L = self.canv_W - 2 * self.slider_x
        y = self.slider_y
        x = self.slider_x + pos * L
        id_outer = self.canv.create_oval(
            x - R,
            y - R,
            x + R,
            y + R,
            fill=Slider.BAR_COLOR_OUTTER,
            width=2,
            outline="",
        )
        id_inner = self.canv.create_oval(
            x - r, y - r, x + r, y + r, fill=Slider.BAR_COLOR_INNER, outline=""
        )
        if self.show_value:
            y_value = y + Slider.BAR_RADIUS + 8
            value = pos * (self.max_val - self.min_val) + self.min_val
            id_value = self.canv.create_text(
                x, y_value, text=format(value, Slider.DIGIT_PRECISION)
            )
            return [id_outer, id_inner, id_value]
        else:
            return [id_outer, id_inner]

    def __moveBar(self, idx, pos):
        ids = self.bars[idx]["Ids"]
        for id in ids:
            self.canv.delete(id)
        self.bars[idx]["Ids"] = self.__addBar(pos)
        self.bars[idx]["Pos"] = pos
        self.bars[idx]["Value"] = pos * (self.max_val - self.min_val) + self.min_val

    def __calcPos(self, x):
        """calculate position from x coordinate"""
        pos = (x - self.slider_x) / (self.canv_W - 2 * self.slider_x)
        if pos < 0:
            return 0
        elif pos > 1:
            return 1
        else:
            return pos

    def __getValue(self, idx):
        """#######Not used function#####"""
        bar = self.bars[idx]
        ids = bar["Ids"]
        x = self.canv.coords(ids[0])[0] + Slider.BAR_RADIUS
        pos = self.__calcPos(x)
        return pos * (self.max_val - self.min_val) + self.min_val

    def __checkSelection(self, x, y):
        """
        To check if the position is inside the bounding rectangle of a Bar
        Return [True, bar_index] or [False, None]
        """
        for idx in range(len(self.bars)):
            id = self.bars[idx]["Ids"][0]
            bbox = self.canv.bbox(id)
            if bbox[0] < x and bbox[2] > x and bbox[1] < y and bbox[3] > y:
                return [True, idx]
        return [False, None]


if __name__ == '__main__':
    app = App()
    app.mainloop()
