import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from pykato.plotfunction.gridspec_layout import Image_Colorbar_Layout, Image_Colorbar_Colorbar_Layout
from pykato.plotfunction.diagram import Image_Colorbar_Diagram, Complex_Diagram
import matplotlib.pyplot as plt
import numpy as np
from pykato.function import Gauss2d, vortex, polka



class MyMainWindow(QMainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()

        gauss_data = Gauss2d((200, 200), offset=0, height=1, width=(25, 25), center=(100, 100), tilt=0)
        polka_data = polka((200, 200), radius=5, spacing=(25,25), offset=(0,0))
        vortex_data = (2 * vortex((200, 200), 4) - 1) * np.pi
        complex_data = polka_data*np.exp(1j*vortex_data)

        self.image_colorbar_layout_figure = Image_Colorbar_Layout()
        Image_Colorbar_Diagram(gauss_data, self.image_colorbar_layout_figure)
        image_colorbar_layout_canvas = FigureCanvas(self.image_colorbar_layout_figure)

        self.image_colorbar_colorbar_layout_figure = Image_Colorbar_Colorbar_Layout()
        Complex_Diagram(complex_data, self.image_colorbar_colorbar_layout_figure)
        image_colorbar_colorbar_layout_canvas = FigureCanvas(self.image_colorbar_colorbar_layout_figure)

        # Create the central widget
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        # Create a layout for the central widget
        vBoxLayout = QVBoxLayout(centralWidget)

        # Create a tab widget
        tabWidget = QTabWidget(self)

        # Create tab and add them to the tab widget
        image_colorbar_layout_tab = QWidget()
        tabWidget.addTab(image_colorbar_layout_tab, "Image_Colorbar_Layout()")
        self.add_content_to_tab(image_colorbar_layout_tab, image_colorbar_layout_canvas)

        image_colorbar_colorbar_layout_tab = QWidget()
        tabWidget.addTab(image_colorbar_colorbar_layout_tab, "Image_Colorbar_Colorbar_Layout()")
        self.add_content_to_tab(image_colorbar_colorbar_layout_tab, image_colorbar_colorbar_layout_canvas)

        # Add the tab widget to the layout
        vBoxLayout.addWidget(tabWidget)
        vBoxLayout.addWidget(NavigationToolbar(image_colorbar_layout_canvas, self))

    def resizeEvent(self, event):
        self.image_colorbar_layout_figure.resize(event)
        self.image_colorbar_colorbar_layout_figure.resize(event)

    def add_content_to_tab(self, tab, widget):
        layout = QVBoxLayout(tab)
        layout.addWidget(widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
