import sys
import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import qdarkstyle

from src.clipig.app.mainwindow import MainWindow
from src.clipig.clipig_worker import ClipigWorker


def main():
    app = QApplication(sys.argv)

    app.setStyleSheet(qdarkstyle.load_stylesheet())
    screen = app.primaryScreen()

    QLocale.setDefault(QLocale("en"))

    win = MainWindow(clipig=ClipigWorker())
    # app.aboutToQuit.connect(win.slot_save_sessions)

    win.showMaximized()
    win.setGeometry(screen.availableGeometry())

    with win.clipig:
        result = app.exec_()

    sys.exit(result)
