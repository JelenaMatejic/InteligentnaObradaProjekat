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
        self.canvas = None
        self.window = window
        self.box = Entry(window)
        self.buttonPerceptron = Button (window, text="Perceptron", command=self.plotPerceptron)
        self.buttonLogisticRegression = Button (window, text="Logistic Regression", command=self.plotLogisticRegression)
        self.buttonMulticlassOneVsAllPerceptron = Button(window, text="One Vs All - Perceptron", command=self.plotMulticlassOneVsAllPerceptron)
        self.buttonMulticlassOneVsAllLogisticRegression = Button(window, text="One Vs All - Logistic Regression",
                                                                 command=self.plotMulticlassOneVsAllLogisticRegression)
        self.buttonPerceptron.grid(row=1, column=0)
        self.buttonMulticlassOneVsAllPerceptron.grid(row=1, column=1)
        self.buttonLogisticRegression.grid(row=2, column=0)
        self.buttonMulticlassOneVsAllLogisticRegression.grid(row=2, column=1)

    def plotPerceptron (self):
        fig = Perceptron.perceptronPlotInWindow()
        self.drawFigure(fig)

    def drawFigure(self, fig):
        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(fig, master=self.window)
            self.canvas.get_tk_widget().grid(row = 3, columnspan=2)
            self.canvas.draw()
        else:
            self.canvas.get_tk_widget().destroy()
            self.canvas = FigureCanvasTkAgg(fig, master=self.window)
            self.canvas.get_tk_widget().grid(row = 3, columnspan=2)
            self.canvas.draw()

    def plotLogisticRegression(self):
        fig = LogisticRegression.logisticRegressionPlotInWindow()
        self.drawFigure(fig)

    def plotMulticlassOneVsAllPerceptron(self):
        fig = MulticlassOneVsAll.oneVsAllPerceptronExample()
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def plotMulticlassOneVsAllLogisticRegression(self):
        fig = MulticlassOneVsAll.oneVsAllLogisticRegressionExample()
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()

window= Tk()
windowClass (window)
window.geometry("650x600")
window.configure(background='white')
window.title("Classifiers")
window.mainloop()
window.quit()