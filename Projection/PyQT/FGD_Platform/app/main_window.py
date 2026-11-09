from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame, QButtonGroup, QStackedWidget)
from app.pages.history_data_page import HistoryDataPage
from app.pages.online_data_page import OnlineDataPage
from app.pages.model_page import ModelPage

# 主菜单
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('数据平台')
        self.resize(1400, 750)

        self.setStyleSheet('background-color: #121417;')

        # 1. 中心部件和全局水平布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 2. 创建左侧导航栏面板
        sidebar = QFrame()
        sidebar.setFixedWidth(210)
        sidebar.setStyleSheet("""
            QFrame {
                background-color : #FFFFFF;
                border-right:1px solid #EFEFEF;
            }
        """)

        # 3. 导航栏内部使用垂直布局
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 20, 12, 16)
        side_layout.setSpacing(6)

        # 4. 标题标签
        title = QLabel('FGD 工控系统')
        title.setStyleSheet("""
            QLabel {
                color: #2F3437;
                font-size: 16px;
                font-weight: 700;
                font-family: 'Segoe UI', 'Microsoft YaHei';
                padding-left: 8px;
                border: none;
            }
        """)

        subtitle = QLabel('DATA & MONITORING')
        subtitle.setStyleSheet("""
            QLabel {
                color: #9A9A97;
                font-size: 10px;
                font-weight: 600;
                font-family: 'Segoe UI';
                padding-left: 8px;
                letter-spacing: 1px;
                border: none;
            }
        """)

        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(18)  # 标题与下方按钮群拉开视觉距离

        # 5. 添加菜单按钮
        btn_history_data = QPushButton('历史数据')
        btn_online_data = QPushButton('在线数据')
        btn_model = QPushButton('模型')
        btn_group = QButtonGroup(self)

        button_style = """
            QPushButton {
                background-color: transparent;
                color: #ACABA9;
                border: none;
                border-left: 3px solid transparent;
                border-radius: 6px;
                padding: 10px;
                text-align: left;
                font: bold 14px "Microsoft YaHei";
            }
            QPushButton:hover {
                background-color: #EFEDEA;
                color: #37352F;
            }
            QPushButton:checked {
                background-color: #EAEAE7;
                color: #37352F;
                border-left: 3px solid #37352F; /* 激活时左侧蓝色指示条 */
                border-top-left-radius: 0px;
                border-bottom-left-radius: 0px;
            }
        """
        for btn in [btn_history_data, btn_online_data, btn_model]:
            btn.setStyleSheet(button_style)
            btn.setCheckable(True)
            side_layout.addWidget(btn)
            btn_group.addButton(btn)

        btn_history_data.setChecked(True)
        side_layout.addStretch()

        # 6. 底部系统状态栏
        status_bar = QLabel('●  离线模式')
        status_bar.setStyleSheet("""
            QLabel {
                color: #0F7B6C; /* 柔和的工业安全绿 */
                font-size: 11px;
                font-weight: bold;
                padding-left: 8px;
                border: none;
            }
        """)
        side_layout.addWidget(status_bar)
        main_layout.addWidget(sidebar)

        # 7. 创建右侧堆叠容器
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(
            'background-color: #F7F6F3;'
        )
        self.page_history = HistoryDataPage()
        self.page_online = OnlineDataPage()
        self.page_model = ModelPage()

        self.stack.addWidget(self.page_history)
        self.stack.addWidget(self.page_online)
        self.stack.addWidget(self.page_model)

        main_layout.addWidget(self.stack)

        # 8. 绑定按钮点击与页面切换
        btn_history_data.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_online_data.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_model.clicked.connect(lambda: self.stack.setCurrentIndex(2))