import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Métodos Numéricos')
        self.setGeometry(100, 100, 800, 600)

        # Crear layout
        self.layout = QVBoxLayout()

        # Crear un canvas de Matplotlib
        self.canvas = FigureCanvas(Figure(figsize=(5, 4), dpi=100))
        self.layout.addWidget(self.canvas)

        # Crear un botón
        self.button = QPushButton('Generar Gráfico')
        self.button.clicked.connect(self.plot_graph)
        self.layout.addWidget(self.button)

        # Configurar el layout
        self.setLayout(self.layout)

    def plot_graph(self):
        # Función para mostrar un gráfico simple de un método numérico (por ejemplo, método de bisección)
        x = np.linspace(-10, 10, 100)
        y = np.sin(x)  # Usamos sin(x) como ejemplo

        ax = self.canvas.figure.add_subplot(111)
        ax.plot(x, y)
        ax.set_title('Gráfico de sin(x)')

        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
