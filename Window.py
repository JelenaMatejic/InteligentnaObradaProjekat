import matplotlib

import MulticlassOneVsAll
from Algorithms import PassiveAggressiveAlgorithm, LogisticRegression, Perceptron
from Plotting import Plot, PlottingAlgorithm, PlotInWindow

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
        self.labelForFile = Label(window, text = "No file chosen . . .", bg = "white", height=1, width=50,  anchor="e", font=("Helvetica", 11))
        self.buttonChooseFile = Button(window, text="Choose Fle", command=self.openFileDialog, font=("Helvetica", 11))
        self.buttonPerceptron = Button (window, text="Perceptron Algorithm", command=self.plotPerceptron, height = 1, width = 50, font=("Helvetica", 11))
        self.buttonLogisticRegression = Button (window, text="Logistic Regression Algorithm", command=self.plotLogisticRegression, height = 1, width = 50, font=("Helvetica", 11))
        self.buttonPassiveAggressive = Button (window, text="Passive Aggressive Algorithm", command=self.plotPassiveAggressive, height = 1, width = 50, font=("Helvetica", 11))
        self.buttonMulticlassOneVsAllPerceptron = Button(window, text="One Vs All - Perceptron Algorithm", command=self.plotMulticlassOneVsAllPerceptron, height = 1, width = 50, font=("Helvetica", 11))
        self.buttonMulticlassOneVsAllLogisticRegression = Button(window, text="One Vs All - Logistic Regression algorithm",
                                                                 command=self.plotMulticlassOneVsAllLogisticRegression, height = 1, width = 50, font=("Helvetica", 11))
        self.buttonPassiveAggressiveOneVsAllLogisticRegression = Button(window, text="One Vs All - Passive Aggressive algorithm",
                                                                 command=self.plotMulticlassOneVsAllPassiveAggressive, height=1, width=50, font=("Helvetica", 11))
        self.labelTitle = Label(window, text = "", bg = "white", font=("Helvetica", 14), pady = 5)

        self.labelForFile.grid(row=0, column=0)
        self.buttonChooseFile.grid(row=0, column=1, sticky=W)
        self.buttonPerceptron.grid(row=1, column=0)
        self.buttonLogisticRegression.grid(row=2, column=0)
        self.buttonPassiveAggressive.grid(row=3, column=0)
        self.buttonMulticlassOneVsAllPerceptron.grid(row=1, column=1)
        self.buttonMulticlassOneVsAllLogisticRegression.grid(row=2, column=1)
        self.buttonPassiveAggressiveOneVsAllLogisticRegression.grid(row=3, column=1)
        self.labelTitle.grid(row=4, columnspan=2)

    def openFileDialog (self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                     filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
        self.labelForFile['text'] = self.filename

    def drawFigure(self, fig):
        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(fig, master=self.window)
            self.canvas.get_tk_widget().grid(row = 5, columnspan=2)
            self.canvas.draw()
        else:
            self.canvas.get_tk_widget().destroy()
            self.canvas = FigureCanvasTkAgg(fig, master=self.window)
            self.canvas.get_tk_widget().grid(row = 5, columnspan=2)
            self.canvas.draw()
        PlotInWindow.fig = Figure()

    def plotPerceptron (self):
        PlotInWindow.fig = Figure()
        file = self.labelForFile["text"]
        self.labelTitle["text"] = "Perceptron Algorithm"
        if file == "" or file == "No file chosen . . .":
            file = "./dataSets/PerceptronDataSet.txt"
        fig = Perceptron.perceptronPlotInWindow(file)
        self.drawFigure(fig)

    def plotLogisticRegression(self):
        PlotInWindow.fig = Figure()
        file = self.labelForFile["text"]
        self.labelTitle["text"] = "Logistic Regression Algorithm"
        if file == "" or file == "No file chosen . . .":
            file = "./dataSets/LogisticRegressionDataSet.txt"
        fig = LogisticRegression.logisticRegressionPlotInWindow(file)
        self.drawFigure(fig)

    def plotPassiveAggressive(self):
        PlotInWindow.fig = Figure()
        file = self.labelForFile["text"]
        self.labelTitle["text"] = "Passive Aggresive Algorithm"
        if file == "" or file == "No file chosen . . .":
            file = "./dataSets/PerceptronDataSet.txt"
        fig = PassiveAggressiveAlgorithm.passiveAggressivePlotInWindow(file)
        self.drawFigure(fig)

    def plotMulticlassOneVsAllPerceptron(self):
        PlotInWindow.fig = Figure()
        file = self.labelForFile["text"]
        self.labelTitle["text"] = "Multiclass One Vs ALL - Perceptron Algorithm"
        if file == "" or file == "No file chosen . . .":
            file = "./dataSets/MulticlassOneVsAllDataSet.txt"
        fig = MulticlassOneVsAll.oneVsAllPerceptronExample(file)
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().grid(row = 5, columnspan=2)
        canvas.draw()

    def plotMulticlassOneVsAllLogisticRegression(self):
        PlotInWindow.fig = Figure()
        file = self.labelForFile["text"]
        self.labelTitle["text"] = "Multiclass One Vs ALL -Logistic Regression Algorithm"
        if file == "" or file == "No file chosen . . .":
            file = "./dataSets/MulticlassOneVsAllDataSet.txt"
        fig = MulticlassOneVsAll.oneVsAllLogisticRegressionExample(file)
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().grid(row = 5, columnspan=2)
        canvas.draw()

    def plotMulticlassOneVsAllPassiveAggressive(self):
        PlotInWindow.fig = Figure()
        file = self.labelForFile["text"]
        self.labelTitle["text"] = "Multiclass One Vs ALL -Logistic Regression Algorithm"
        if file == "" or file == "No file chosen . . .":
            file = "./dataSets/MulticlassOneVsAllDataSet.txt"
        fig = MulticlassOneVsAll.oneVsAllPassiveAggressiveExample(file)
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().grid(row = 5, columnspan=2)
        canvas.draw()
window= Tk()
windowClass (window)
window.geometry("920x620")
window.configure(background='white')
window.title("Classifiers")
window.mainloop()
window.quit()