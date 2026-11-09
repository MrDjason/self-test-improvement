from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

# 2.在线数据页面
class OnlineDataPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel('实时数据监控')
        title.setStyleSheet(
            'color: #2F3437; font-size: 18px; font-weight: bold; border: none;'
        )
        layout.addWidget(title)
        layout.addStretch()