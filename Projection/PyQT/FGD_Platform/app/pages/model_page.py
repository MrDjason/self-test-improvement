from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

# 3.控制模型页面
class ModelPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        title = QLabel('模型')
        title.setStyleSheet(
            'color: #2F3437; font-size: 18px; font-weight: bold; border: none;'
        )
        layout.addWidget(title)
        layout.addStretch()
