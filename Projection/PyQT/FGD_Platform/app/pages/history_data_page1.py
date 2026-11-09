import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QPushButton, QTableWidget, QHeaderView, QFrame, QComboBox, QDateTimeEdit, QFileDialog, QMessageBox, QTableWidgetItem, QColorDialog
from PySide6.QtGui import QColor, QPixmap, QIcon
from app.utils.data_loader import DataLoaderThread

# 1. 历史数据页面
class HistoryDataPage(QWidget):
    def __init__(self):
        super().__init__()
        self.raw_data = {}
        self.data_ranges = {}
        self.normalized_01_data = {}
        self.waterfall_data = {}
        self.lane_ranges = {}
        self.lane_dividers = []
        self.curves = {}
        self.init_ui()
        self.palette = [
            '#2383E2', '#E03E3E', '#0F7B6C', '#D9730D', '#6940A5', 
            '#0B6E99', '#AD1A72', '#4D6461', '#DFAB01', '#9065B0'
        ]
        self.point_colors = {}

    def _create_color_icon(self, hex_color: str) -> QIcon:
        pixmap = QPixmap(14, 14)
        pixmap.fill(QColor(hex_color))
        return QIcon(pixmap)
    
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
        header.setSectionResizeMode(3, QHeaderView.Fixed); self.table.setColumnWidth(3, 150) # 数值范围（固定100）
        
        # 右侧四列
        header.setSectionResizeMode(4, QHeaderView.Fixed); self.table.setColumnWidth(4, 40)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Fixed); self.table.setColumnWidth(6, 60)
        header.setSectionResizeMode(7, QHeaderView.Fixed); self.table.setColumnWidth(7, 150)

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


        # 监听用户勾选改变事件
        self.table.itemChanged.connect(self.on_table_item_changed)
        # 监听表格双击事件 用于双击颜色单元格时弹出调色板
        self.table.cellDoubleClicked.connect(self.on_color_cell_double_clicked)

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

        # 阻止信号触发多次重算
        self.table.blockSignals(True)

        # 填充表格
        total_points = len(valid_cols)
        half_rows = (total_points + 1) // 2
        self.table.setRowCount(half_rows)
        self.table.clearContents()

        for r, col in enumerate(valid_cols):
            # 判断左栏还是右栏
            row_idx = r if r < half_rows else r - half_rows
            col_offset = 0 if r < half_rows else 4

            # 初始分配颜色
            color_hex = self.palette[r % len(self.palette)]
            self.point_colors[col] = color_hex            

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

            # 颜色单元格
            color_item = QTableWidgetItem(self._create_color_icon(color_hex), ' 选择')
            color_item.setTextAlignment(Qt.AlignCenter)
            color_item.setData(Qt.UserRole, col)
            color_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row_idx, col_offset + 2, color_item)

            # 工程量范围
            v_min, v_max = np.nanmin(numeric_data[col]), np.nanmax(numeric_data[col])
            self.table.setItem(row_idx, col_offset + 3, QTableWidgetItem(f'{v_min:.2f} ~ {v_max:.2f}'))
        
        self.table.blockSignals(False)

        # 依次执行规整与泳道排布
        self.process_data_normalization()
        self.rebuild_waterfall_layout()

        if self.timestamps is not None and len(self.timestamps) > 1:
            x_min = float(self.timestamps[0])
            x_max = float(self.timestamps[-1])
            vb = self.plot_widget.getViewBox()
            vb.setRange(xRange=(x_min, x_max), yRange=(0, 100), padding = 0.01)

    # 响应颜色单元格双击：唤起系统调色盘并实时回写表格和图表
    def on_color_cell_double_clicked(self, row, col):
        # 仅响应两栏布局下的颜色列：第 2 列（左栏颜色）和第 6 列（右栏颜色）
        if col not in (2, 6):
            return

        item = self.table.item(row, col)
        if not item:
            return

        col_name = item.data(Qt.UserRole)
        if not col_name:
            return

        # 调起系统调色盘
        initial_color = QColor(self.point_colors.get(col_name, '#2383E2'))
        chosen_color = QColorDialog.getColor(initial_color, self, f"选择 [{col_name}] 曲线颜色")

        if chosen_color.isValid():
            hex_color = chosen_color.name()
            # 1. 更新映射字典
            self.point_colors[col_name] = hex_color

            # 2. 更新表格内的色块图标
            item.setIcon(self._create_color_icon(hex_color))

            # 3. 联动更新折线图已有曲线的笔刷颜色
            if col_name in self.curves:
                pen = pg.mkPen(color=hex_color, width=1.5)
                self.curves[col_name].setPen(pen)

    # 双段非线性规整 
    def process_data_normalization(self):
        self.data_ranges.clear()
        self.normalized_01_data.clear()

        for col, arr in self.raw_data.items(): # 遍历 列名 和 值
            min_v = float(np.nanmin(arr))
            max_v = float(np.nanmax(arr))
            self.data_ranges[col] = (min_v, max_v) # 保存原始量程
            diff = max_v - min_v
            res = np.zeros_like(arr, dtype=np.float32)

            threshold = min_v + max(0.1, diff*0.01)

            non_zero_mask = arr > threshold # 逐元素比较得到bool数组

            if np.any(non_zero_mask): # 全停机时跳过
                nz_min = float(np.nanmin(arr[non_zero_mask])) # 获取运行段（非停机）最小值
                nz_diff = max_v - nz_min # 运行段的跨度用全局最大值减去运行段最小值
                if nz_diff > 1e-4:
                    res[non_zero_mask] = 0.15 + ((arr[non_zero_mask] - nz_min) / nz_diff) * 0.80
                else:
                    res[non_zero_mask] = 0.55

            res[~non_zero_mask] = 0.03 # 留距离防止和x轴重叠
            self.normalized_01_data[col] = res # 储存键值对

    # 根据选择曲线分配泳道
    def rebuild_waterfall_layout(self):
        visible_cols = []
        # 1. 抓取被勾选的测点
        for r in range(self.table.rowCount()):
            for offset in (0, 4):          # 左栏 offset=0，右栏 offset=4
                chk = self.table.item(r, offset)
                name_item = self.table.item(r, offset + 1)
                if chk and name_item and chk.checkState() == Qt.Checked:
                    col_name = name_item.data(Qt.UserRole)
                    if col_name and col_name in self.normalized_01_data:
                        visible_cols.append(col_name)

        # 2. 清理旧分割虚线
        for line in self.lane_dividers:
            self.plot_widget.removeItem(line)
        self.lane_dividers.clear()
        self.lane_ranges.clear()

        # 3. 清理已取消勾选的曲线
        for col in list(self.curves.keys()):
            if col not in visible_cols:
                self.plot_widget.removeItem(self.curves[col])
                del self.curves[col]

        n = len(visible_cols)
        if n == 0:
            self.waterfall_data.clear()
            return

        # 4. 计算垂直通道分层
        lane_height = 96.0 / n
        gap = lane_height * 0.12

        for i, col in enumerate(visible_cols):
            lane_top = 98.0 - i * lane_height
            lane_bottom = lane_top - (lane_height - gap)
            self.lane_ranges[col] = (lane_bottom, lane_top)

            # 将规整化值映射到分配的泳道高度
            norm_arr = self.normalized_01_data[col]
            self.waterfall_data[col] = lane_bottom + norm_arr * (lane_top - lane_bottom)

            # 绘制不同测点之间的水平分割线
            if i > 0:
                div_line = pg.InfiniteLine(pos = lane_top + gap / 2.0, angle = 0, pen = pg.mkPen('#787774', style=Qt.DashLine))
                self.plot_widget.addItem(div_line)
                self.lane_dividers.append(div_line)

            # 上屏绘制
            if col not in self.curves:
                pen = pg.mkPen(color='#2383E2', width=1.5)
                self.curves[col] = self.plot_widget.plot(
                    x = self.timestamps,
                    y = self.waterfall_data[col],
                    pen = pen
                )
            else:
                self.curves[col].setData(x=self.timestamps, y=self.waterfall_data[col])

    def on_table_item_changed(self, item):
        if item.column() in (0, 4):
            self.rebuild_waterfall_layout()