import Perceptron
import LogisticRegression
import Plot
import sys
import MulticlassOneVsAll
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import filedialog
from matplotlib.figure import Figure

# root.title("Multiclass One VS All")
# perceptronButton = Button(root, text = "Perceptron example", command = Perceptron.perceptronExample).pack()
# root.mainloop()

class windowClass:

    def __init__(self,  window):
        self.canvas = None
        self.window = window
        self.box = Entry(window)
        # self.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
        #                                              filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.labelForFile = Label(window, text = "No file chosen . . .", bg = "white", height=1, width=50,  anchor="e")
        self.buttonChooseFile = Button(window, text="Choose Fle")
        self.buttonPerceptron = Button (window, text="Perceptron Algorithm", command=self.plotPerceptron, height = 1, width = 50)
        self.buttonLogisticRegression = Button (window, text="Logistic Regression Algorithm", command=self.plotLogisticRegression, height = 1, width = 50)
        self.buttonPassiveAggressive = Button (window, text="Passive Aggressive Algorithm", command=self.plotLogisticRegression, height = 1, width = 50)
        self.buttonMulticlassOneVsAllPerceptron = Button(window, text="One Vs All - Perceptron Algorithm", command=self.plotMulticlassOneVsAllPerceptron, height = 1, width = 50)
        self.buttonMulticlassOneVsAllLogisticRegression = Button(window, text="One Vs All - Logistic Regression algorithm",
                                                                 command=self.plotMulticlassOneVsAllLogisticRegression, height = 1, width = 50)
        self.buttonPassiveAggressiveOneVsAllLogisticRegression = Button(window, text="One Vs All - Logistic Regression algorithm",
                                                                 command=self.plotMulticlassOneVsAllLogisticRegression, height=1, width=50)
        self.labelForFile.grid(row=0, column=0)
        self.buttonChooseFile.grid(row=0, column=1, sticky=W)
        self.buttonPerceptron.grid(row=1, column=0)
        self.buttonLogisticRegression.grid(row=2, column=0)
        self.buttonPassiveAggressive.grid(row=3, column=0)
        self.buttonMulticlassOneVsAllPerceptron.grid(row=1, column=1)
        self.buttonMulticlassOneVsAllLogisticRegression.grid(row=2, column=1)
        self.buttonPassiveAggressiveOneVsAllLogisticRegression.grid(row=3, column=1)

    def plotPerceptron (self):
        fig = Perceptron.perceptronPlotInWindow()
        self.drawFigure(fig)

    def drawFigure(self, fig):
        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(fig, master=self.window)
            self.canvas.get_tk_widget().grid(row = 4, columnspan=2)
            self.canvas.draw()
        else:
            self.canvas.get_tk_widget().destroy()
            self.canvas = FigureCanvasTkAgg(fig, master=self.window)
            self.canvas.get_tk_widget().grid(row = 4, columnspan=2)
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
window.geometry("720x600")
window.configure(background='white')
window.title("Classifiers")
#filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
#print(filename)
window.mainloop()
window.quit()