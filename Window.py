import Perceptron
import Plot
import sys
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *


# root.title("Multiclass One VS All")
# perceptronButton = Button(root, text = "Perceptron example", command = Perceptron.perceptronExample).pack()
# root.mainloop()

class windowClass:

    def __init__(self,  window):
        self.window = window
        self.box = Entry(window)
        self.buttonPerceptron = Button (window, text="Perceptron", command=self.plot)
        self.buttonPerceptron.pack()

    def plot (self):
        fig = Perceptron.perceptronPlotInWindow()
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()

window= Tk()
windowClass (window)
window.geometry("500x500")
window.title("Classifiers")
window.mainloop()