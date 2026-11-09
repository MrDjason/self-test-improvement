from PySide6.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QPushButton
# QApplication   应用程序对象
# QMainWindow    主窗口
# QPushButton    按钮
# QPlainTextEdit 多行文本编辑框

def handleCalc():
    print('统计按钮被点击了')

app = QApplication() # GUI应用程序对象，管理事件循环、窗口系统

window = QMainWindow()  # 创建主窗口对象
window.resize(500, 400) # 设置窗口大小
window.move(300,310)    # 窗口移动到屏幕坐标(300, 310)处
window.setWindowTitle('第一个窗口')

textEdit = QPlainTextEdit(window) # 把文本框作为窗口的子控件
textEdit.setPlaceholderText('姓名') # 无文字时灰色提示文字
textEdit.move(10, 25)
textEdit.resize(300, 350)

button = QPushButton('统计', window)
button.move(380, 80)
button.clicked.connect(handleCalc)

window.show() # 展示窗口
app.exec()    # 事件循环