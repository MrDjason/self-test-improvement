import sys
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

pg.setConfigOption('background', '#FFFFFF')
pg.setConfigOption('foreground', '#5F5E5B')
pg.setConfigOption('antialias', True)

from app.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
    