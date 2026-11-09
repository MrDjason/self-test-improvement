import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from datetime import datetime
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QDateTime
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, \
QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame, QButtonGroup, \
QStackedWidget, QFileDialog, QMessageBox, QComboBox, QTableWidget, \
QTableWidgetItem, QHeaderView, QScrollArea, QColorDialog, QDateTimeEdit

pg.setConfigOption('background', '#FFFFFF')
pg.setConfigOption('foreground', '#5F5E5B')
pg.setConfigOption('antialias', True)

# 主菜单
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('数据平台')
        self.resize(1000, 650)

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

# 1. 历史数据页面
class HistoryDataPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)

        # 页面滚动
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) # 自动拉伸
        scroll_area.setStyleSheet('QScrollArea { border: none; background-color: transparent; }')

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(24, 24, 24, 24) # 页面边距
        layout.setSpacing(12)                     # 页面控件之间垂直间距

        # 1. 顶部控制栏
        top_bar = QHBoxLayout()
        title = QLabel('历史数据分析')
        title.setStyleSheet('color: #2F3437; font-size: 18px; font-weight: bold; border: none;')
        top_bar.addWidget(title)

        self.btn_import = QPushButton('导入 CSV')
        self.btn_import.setStyleSheet('background-color: #2383E2; color: #FFF; padding: 4px 10px; font-weight: bold; border-radius: 4px;')
        self.btn_import.clicked.connect(self.select_file)
        top_bar.addWidget(self.btn_import)

        self.lbl_status = QLabel('等待导入数据...')
        self.lbl_status.setStyleSheet('color: #888888; font-size: 12px;')
        top_bar.addWidget(self.lbl_status)

        top_bar.addStretch()

        layout.addLayout(top_bar) 

        # 2. 测点配置表格 分两栏
        # 左侧表格
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            '显示', '测点名称', '颜色', '数值范围',
            '显示', '测点名称', '颜色', '数值范围'
            ])
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(50) # 表头最小尺寸

        # 左侧四列
        header.setSectionResizeMode(0, QHeaderView.Fixed); self.table.setColumnWidth(0, 40) # 显示（固定40）
        header.setSectionResizeMode(1, QHeaderView.Stretch) # 测点名称（自适应拉伸）
        header.setSectionResizeMode(2, QHeaderView.Fixed); self.table.setColumnWidth(2, 60) # 颜色（固定60）
        header.setSectionResizeMode(3, QHeaderView.Fixed); self.table.setColumnWidth(3, 100) # 数值范围（固定100）
        
        # 右侧四列
        header.setSectionResizeMode(4, QHeaderView.Fixed); self.table.setColumnWidth(4, 40)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Fixed); self.table.setColumnWidth(6, 60)
        header.setSectionResizeMode(7, QHeaderView.Fixed); self.table.setColumnWidth(7, 100)

        # 表格美化
        self.table.verticalHeader().setDefaultSectionSize(36) # 行高
        self.table.horizontalHeader().setFixedHeight(38) # 表头高度
        self.table.verticalHeader().setVisible(False) # 取消行序号
        self.table.setAlternatingRowColors(True) # 开启交替行颜色
        
        self.table.setFixedHeight(300)
        self.table.setStyleSheet("""
            /* 1. 基础表格样式 */
            QTableWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F8F9FA;
                border: 1px solid #D0D0D0;
                border-radius: 6px;
                gridline-color: #E8E8E8;
                outline: none;
            }
            
            /* 2. 阻止单元格被选中时变灰 */
            QTableWidget::item:selected {
                background-color: transparent;
                color: #2F3437;
            }
            
            /* 3. 表头样式 */
            QHeaderView::section {
                background-color: #F7F8FA;
                color: #555555;
                font-weight: bold;
                border: none;
                border-bottom: 2px solid #E0E0E0;
                border-right: 1px solid #E8E8E8;
                padding: 6px;
            }
            
            /* 4. 复选框变成纯色块 */
            /* 未选中：白底灰边 */
            QTableWidget::indicator:unchecked {
                background-color: #FFFFFF;
                border: 1px solid #999999;
                border-radius: 3px;
            }
            /* 选中：纯亮蓝底亮蓝边，没有任何图片干扰 */
            QTableWidget::indicator:checked {
                background-color: #2383E2;
                border: 1px solid #2383E2;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.table)

        # 3. PyQtGraph 图表区域
        chart_card = QFrame()
        chart_card.setMinimumHeight(450) # 最小高度
        chart_card.setStyleSheet("QFrame { background-color: #FFF; border: 1px solid #C4C4C0; border-radius: 6px; }")
        card_layout = QVBoxLayout(chart_card)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.4)

        card_layout.addWidget(self.plot_widget)
        layout.addWidget(chart_card)

        # 画布固定
        vb = self.plot_widget.getViewBox()
        vb.setMouseMode(pg.ViewBox.PanMode)
        vb.setMouseEnabled(x=True, y=False)
        vb.setYRange(0, 100, padding=0.02)

        # 底部栏
        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(QLabel('时间跨度：'))
        self.combo_span = QComboBox()
        self.combo_span.addItems(['全量', '10分钟', '1小时', '6小时', '12小时', '24小时'])
        bottom_bar.addWidget(self.combo_span)

        bottom_bar.addWidget(QLabel('定位时间点：'))
        self.dt_picker = QDateTimeEdit()
        self.dt_picker.setDisplayFormat('yyyy-MM-dd HH:mm:ss')
        self.dt_picker.setCalendarPopup(True)
        bottom_bar.addWidget(self.dt_picker)

        bottom_bar.addStretch()
        layout.addLayout(bottom_bar)

        # 收尾
        scroll_area.setWidget(content_widget)
        page_layout.addWidget(scroll_area)

    def select_file(self):
        # 1. 弹窗
        path, _ = QFileDialog.getOpenFileName(self, '选择CSV', '', 'CSV Files (*.csv)')
        if not path:
            return # 点击取消直接退出

        # 2. 更新 UI 状态
        self.lbl_status.setText('正在解析...')

        # 3. 启动后台线程
        self.thread = DataLoaderThread(path)

        # 4. 绑定信号
        self.thread.data_loaded.connect(self.on_data_ready)
        self.thread.error_occurred.connect(lambda err:QMessageBox.critical(self, '错误', err))

        # 5.运行
        self.thread.start()

    def on_data_ready(self, timestamps, numeric_data, valid_cols):
        self.timestamps = timestamps
        self.raw_data = numeric_data
        self.lbl_status.setText(f'已载入 {len(timestamps)} 行数据')

        # 填充表格
        total_points = len(valid_cols)
        half_rows = (total_points + 1) // 2
        self.table.setRowCount(half_rows)
        self.table.clearContents()

        for r, col in enumerate(valid_cols):
            # 判断左栏还是右栏
            row_idx = r if r < half_rows else r-half_rows
            col_offset = 0 if r < half_rows else 4

            # 勾选框
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if r < 3 else Qt.Unchecked)
            chk.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, col_offset + 0, chk)

            # 测点名
            name_item = QTableWidgetItem(col)
            name_item.setData(Qt.UserRole, col)
            self.table.setItem(row_idx, col_offset + 1, name_item)

            # 颜色
            self.table.setItem(row_idx, col_offset + 2, QTableWidgetItem('■ 颜色'))

            # 工程量范围
            v_min, v_max =np.nanmin(numeric_data[col]), np.max(numeric_data[col])
            self.table.setItem(row_idx, col_offset + 3, QTableWidgetItem(f'{v_min:.2f} ~ {v_max:.2f}'))
        
        self.draw_plots()

    def draw_plots(self):
        self.plot_widget.clear()
        colors = ['#FF5252', '#448AFF', '#4CAF50', '#FFC107', '#9C27B0', '#00BCD4']
        
        for r in range(self.table.rowCount()):
            # 检查左栏（0-3列）
            if self.table.item(r, 0) and self.table.item(r, 0).checkState() == Qt.Checked:
                col_name = self.table.item(r, 1).data(Qt.UserRole)
                self.plot_widget.plot(self.timestamps, self.raw_data[col_name], 
                                      pen=pg.mkPen(color=colors[r % len(colors)], width=2), name=col_name)
            # 检查右栏（4-7列）
            if self.table.item(r, 4) and self.table.item(r, 4).checkState() == Qt.Checked:
                col_name = self.table.item(r, 5).data(Qt.UserRole)
                self.plot_widget.plot(self.timestamps, self.raw_data[col_name], 
                                      pen=pg.mkPen(color=colors[(r + 3) % len(colors)], width=2), name=col_name)


# 异步csv加载
class DataLoaderThread(QThread):
    data_loaded = Signal(object, object, list)
    error_occurred = Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            df = pd.read_csv(self.file_path)
            date_col = 'date' if 'date' in df.columns else df.columns[0] 

            # 转换时间戳序列并排序
            dt_series = pd.to_datetime(df[date_col], errors='coerce').bfill().ffill() # 将时间字符串转为pandas 如果非时间则将至变为 NaT 然后用 bfill().ffill() 用后面或者前面值填补
            timestamps = (dt_series.astype('int64') // 10 ** 9).to_numpy() # 将 pandas 存储的纳秒整除10**9变为秒并转为numpy 计算更快

            sort_order = np.argsort(timestamps) # 找出从小到大排序 原数组元素的索引数组
            timestamps = timestamps[sort_order] # 根据提供的索引数组 从原数组依次取出对应位置的元素 组成新的数组

            numeric_data = {} # 数据
            valid_cols = []   # 列名

            for col in df.columns:
                if col!= date_col and np.issubdtype(df[col].dtype, np.number): # 排除时间列且必须是数字类型
                    col_data = df[col].fillna(0).to_numpy(dtype=np.float32) # float64转为float32 为计算节省内存
                    numeric_data[col] = col_data[sort_order] # 将列表数据随时间大小重新排列
                    valid_cols.append(col)

            self.data_loaded.emit(timestamps, numeric_data, valid_cols)

        except Exception as e:
            self.error_occurred.emit(str(e))

    

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
