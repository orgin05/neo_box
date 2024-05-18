from PyQt5.QtGui import QMouseEvent
from mainwindow_ui import *
from exer_ui import *
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


class LoginWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.ui = Ui_MainWindow()
    self.ui.setupUi(self)
    self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
    self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
    self.ui.pushButton_exer.clicked.connect(self.go_to_exer)
    self.show()

  def go_to_exer(self):
    flag = True
    if(flag):
      self.win = ExerWindow()
      self.close()
    else:
      pass

  def mousePressEvent(self, event):
    if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
      self.m_flag = True
      self.m_Position = event.globalPos() - self.pos()
      event.accept()
      self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))

  def mouseMoveEvent(self, mouse_event):
    if QtCore.Qt.LeftButton and self.m_flag:
      self.move(mouse_event.globalPos() - self.m_Position)
      mouse_event.accept()

  def mouseReleaseEvent(self, mouse_event):
    self.m_flag = False
    self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))


class ExerWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.ui = Ui_MainWindow_exer()
    self.ui.setupUi(self)
    self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
    self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
    self.ui.pushButton_max.clicked.connect(self.resize_win)
    self.ui.pushButton_back.clicked.connect(self.go_to_login)
    self.show()

  def go_to_login(self):
    flag = 1
    if(flag):
      self.win = LoginWindow()
      self.close()
    else:
      pass
    
  def resize_win(self):
    if self.isMaximized():
      self.showNormal()
      self.ui.pushButton_max.setIcon(QtGui.QIcon(u":/icons/icons/maxsize.png"))
    else:
      self.showMaximized()
      self.ui.pushButton_max.setIcon(QtGui.QIcon(u":/icons/icons/minsize.png"))

  def mousePressEvent(self, event):
    if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
      self.m_flag = True
      self.m_Position = event.globalPos() - self.pos()
      event.accept()
      self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))

  def mouseMoveEvent(self, mouse_event):
    if QtCore.Qt.LeftButton and self.m_flag:
      self.move(mouse_event.globalPos() - self.m_Position)
      mouse_event.accept()

  def mouseReleaseEvent(self, mouse_event):
    self.m_flag = False
    self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))


if __name__ == '__main__':
  app = QApplication(sys.argv)
  win = LoginWindow()
  sys.exit(app.exec())
