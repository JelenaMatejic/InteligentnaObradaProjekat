import Perceptron
import LogisticRegression
import Plot
import sys
import MulticlassOneVsAll
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from matplotlib.figure import Figure

# root.title("Multiclass One VS All")
# perceptronButton = Button(root, text = "Perceptron example", command = Perceptron.perceptronExample).pack()
# root.mainloop()

class windowClass:

    def __init__(self,  window):
        self.window = window
        self.box = Entry(window)
        self.buttonPerceptron = Button (window, text="Perceptron", command=self.plotPerceptron)
        self.buttonLogisticRegression = Button (window, text="Logistic Regression", command=self.plotLogisticRegression)
        self.buttonMulticlassOneVsAllPerceptron = Button(window, text="One Vs All - Perceptron", command=self.plotMulticlassOneVsAllPerceptron)
        self.buttonPerceptron.pack()
        self.buttonLogisticRegression.pack()
        self.buttonMulticlassOneVsAllPerceptron.pack()

    def plotPerceptron (self):
        fig = Perceptron.perceptronPlotInWindow()
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def plotLogisticRegression(self):
        fig = LogisticRegression.logisticRegressionPlotInWindow()
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def plotMulticlassOneVsAllPerceptron(self):
        fig = MulticlassOneVsAll.oneVsAllPerceptronExample()
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()

window= Tk()
windowClass (window)
window.geometry("500x500")
window.title("Classifiers")
window.mainloop()