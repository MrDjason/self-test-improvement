import numpy as np
import pyqtgraph as pg
from datetime import datetime
from PySide6.QtCore import Qt, QDateTime, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QPushButton, 
    QTableWidget, QHeaderView, QFrame, QComboBox, QDateTimeEdit, 
    QFileDialog, QMessageBox, QTableWidgetItem, QColorDialog
)
from PySide6.QtGui import QColor, QPixmap, QIcon
from app.utils.data_loader import DataLoaderThread

pg.setConfigOption('antialias', False)


# 双行时间坐标轴（时分秒 / 年月日）
class DualLineTimeAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setHeight(48)
        self._cache = {}

    def tickStrings(self, values, scale, spacing):
        strings = []
        for val in values:
            if np.isnan(val) or val <= 0:
                strings.append('')
                continue
            int_sec = int(val)
            if int_sec in self._cache:
                strings.append(self._cache[int_sec])
            else:
                try:
                    dt = datetime.fromtimestamp(int_sec)
                    formatted = dt.strftime('%H:%M:%S\n%Y-%m-%d')
                    self._cache[int_sec] = formatted
                    strings.append(formatted)
                except Exception:
                    strings.append('')
        
        if len(self._cache) > 2000:
            self._cache.clear()

        return strings


# 纯净版：右侧专属彩色量程标尺（按空间自上而下自绘）
class MultiTrackRightAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWidth(85)
        self.track_configs = []

    def tickValues(self, minVal, maxVal, size):
        return []

    def set_track_configs(self, configs):
        self.track_configs = configs
        self.picture = None
        self.update()

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        if not self.track_configs:
            return

        vb = self.linkedView()
        if vb is None:
            return

        _, (y_min, y_max) = vb.viewRange()
        y_span = max(1e-4, y_max - y_min)
        rect = self.rect()

        for lane_bottom, lane_top, v_min, v_max, hex_color in self.track_configs:
            y_data_top = lane_bottom + 0.92 * (lane_top - lane_bottom)
            y_data_bot = lane_bottom + 0.08 * (lane_top - lane_bottom)

            py_top = rect.bottom() - (y_data_top - y_min) / y_span * rect.height()
            py_bot = rect.bottom() - (y_data_bot - y_min) / y_span * rect.height()

            color = QColor(hex_color)
            p.setPen(color)
            font = p.font()
            font.setPointSize(9)
            font.setBold(True)
            p.setFont(font)

            # 上限短线与数值
            p.drawLine(rect.left(), int(py_top), rect.left() + 6, int(py_top))
            p.drawText(rect.left() + 9, int(py_top) + 4, f"{v_max:.2f}")

            # 下限短线与数值
            p.drawLine(rect.left(), int(py_bot), rect.left() + 6, int(py_bot))
            p.drawText(rect.left() + 9, int(py_bot) + 4, f"{v_min:.2f}")


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
        self.timestamps = None
        self.palette = [
            '#2383E2', '#E03E3E', '#0F7B6C', '#D9730D', '#6940A5', 
            '#0B6E99', '#AD1A72', '#4D6461', '#DFAB01', '#9065B0'
        ]
        self.point_colors = {}
        self.active_focus_col = None
        self.selected_order = []  # 记录用户勾选测点的先后次序

        self.span_seconds_map = {
            '10分钟': 600,
            '1小时': 3600,
            '6小时': 21600,
            '12小时': 43200,
            '24小时': 86400,
            '全量': None
        }

        # 防抖重采样定时器
        self.lod_timer = QTimer(self)
        self.lod_timer.setSingleShot(True)
        self.lod_timer.timeout.connect(self.update_visible_curves_lod)

        self.init_ui()

    def _create_color_icon(self, hex_color: str) -> QIcon:
        pixmap = QPixmap(14, 14)
        pixmap.fill(QColor(hex_color))
        return QIcon(pixmap)

    def init_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet('QScrollArea { border: none; background-color: transparent; }')

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

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

        # 2. 测点配置表格
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            '显示', '测点名称', '颜色', '数值范围',
            '显示', '测点名称', '颜色', '数值范围'
        ])
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(50)

        header.setSectionResizeMode(0, QHeaderView.Fixed); self.table.setColumnWidth(0, 40)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed); self.table.setColumnWidth(2, 70)
        header.setSectionResizeMode(3, QHeaderView.Fixed); self.table.setColumnWidth(3, 150)
        
        header.setSectionResizeMode(4, QHeaderView.Fixed); self.table.setColumnWidth(4, 40)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Fixed); self.table.setColumnWidth(6, 70)
        header.setSectionResizeMode(7, QHeaderView.Fixed); self.table.setColumnWidth(7, 150)

        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.horizontalHeader().setFixedHeight(38)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        
        self.table.setFixedHeight(260)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F8F9FA;
                border: 1px solid #D0D0D0;
                border-radius: 6px;
                gridline-color: #E8E8E8;
                outline: none;
            }
            QTableWidget::item:selected {
                background-color: transparent;
                color: #2F3437;
            }
            QHeaderView::section {
                background-color: #F7F8FA;
                color: #555555;
                font-weight: bold;
                border: none;
                border-bottom: 2px solid #E0E0E0;
                border-right: 1px solid #E8E8E8;
                padding: 6px;
            }
            QTableWidget::indicator:unchecked {
                background-color: #FFFFFF;
                border: 1px solid #999999;
                border-radius: 3px;
            }
            QTableWidget::indicator:checked {
                background-color: #2383E2;
                border: 1px solid #2383E2;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.table)

        # 3. 图表区域容器
        self.chart_card = QFrame()
        self.chart_card.setFixedHeight(480)
        self.chart_card.setStyleSheet("QFrame { background-color: #FFF; border: 1px solid #C4C4C0; border-radius: 6px; }")
        card_layout = QVBoxLayout(self.chart_card)

        custom_time_axis = DualLineTimeAxis(orientation='bottom')
        self.right_axis = MultiTrackRightAxis(orientation='right')

        self.plot_widget = pg.PlotWidget(axisItems={
            'bottom': custom_time_axis,
            'right': self.right_axis
        })
        self.plot_widget.showAxis('right', True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.4)
        self.right_axis.linkToView(self.plot_widget.getViewBox())

        # 游标垂直指示虚线
        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#E03E3E', width=1.5, style=Qt.DashLine))
        self.cursor_line.setVisible(False)
        self.plot_widget.addItem(self.cursor_line, ignoreBounds=True)

        # 浮动指示标尺文本框
        self.cursor_text = pg.TextItem(anchor=(0, 0), fill=pg.mkColor('#FFFFFFEE'), border=pg.mkPen('#999999'))
        self.cursor_text.setVisible(False)
        self.plot_widget.addItem(self.cursor_text, ignoreBounds=True)

        card_layout.addWidget(self.plot_widget)
        layout.addWidget(self.chart_card)

        vb = self.plot_widget.getViewBox()
        vb.setMouseMode(pg.ViewBox.PanMode)
        vb.setMouseEnabled(x=True, y=False)
        vb.setYRange(0, 100, padding=0.02)

        vb.sigRangeChanged.connect(self.on_view_range_changed)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked)

        # 底部栏
        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(QLabel('时间跨度：'))
        self.combo_span = QComboBox()
        self.combo_span.addItems(['10分钟', '1小时', '6小时', '12小时', '24小时', '全量'])
        self.combo_span.setCurrentText('10分钟')
        bottom_bar.addWidget(self.combo_span)

        bottom_bar.addWidget(QLabel('定位时间点：'))
        self.dt_picker = QDateTimeEdit()
        self.dt_picker.setDisplayFormat('yyyy-MM-dd HH:mm:ss')
        self.dt_picker.setCalendarPopup(True)
        bottom_bar.addWidget(self.dt_picker)

        # 标尺清除/隐藏按钮
        self.btn_clear_cursor = QPushButton('隐藏游标')
        self.btn_clear_cursor.setStyleSheet('background-color: #EFEFEF; color: #555; padding: 4px 8px; border-radius: 4px;')
        self.btn_clear_cursor.clicked.connect(self.hide_cursor)
        bottom_bar.addWidget(self.btn_clear_cursor)

        bottom_bar.addStretch()
        layout.addLayout(bottom_bar)

        scroll_area.setWidget(content_widget)
        page_layout.addWidget(scroll_area)

        # 监听事件
        self.table.itemChanged.connect(self.on_table_item_changed)
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.combo_span.currentTextChanged.connect(self.apply_viewport_range)
        self.dt_picker.dateTimeChanged.connect(self.apply_viewport_range)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择CSV', '', 'CSV Files (*.csv)')
        if not path:
            return

        self.lbl_status.setText('正在解析...')
        self.thread = DataLoaderThread(path)
        self.thread.data_loaded.connect(self.on_data_ready)
        self.thread.error_occurred.connect(lambda err: QMessageBox.critical(self, '错误', err))
        self.thread.start()

    def on_data_ready(self, timestamps, numeric_data, valid_cols):
        self.timestamps = timestamps
        self.raw_data = numeric_data
        self.lbl_status.setText(f'已载入 {len(timestamps)} 行数据')
        self.selected_order.clear()

        if len(timestamps) > 0:
            start_sec = int(timestamps[0])
            start_qdt = QDateTime.fromSecsSinceEpoch(start_sec)
            self.dt_picker.blockSignals(True)
            self.dt_picker.setDateTime(start_qdt)
            self.dt_picker.blockSignals(False)

        self.point_colors.clear()
        for idx, col in enumerate(valid_cols):
            self.point_colors[col] = self.palette[idx % len(self.palette)]

        self.table.blockSignals(True)

        total_points = len(valid_cols)
        half_rows = (total_points + 1) // 2
        self.table.setRowCount(half_rows)
        self.table.clearContents()

        for r, col in enumerate(valid_cols):
            row_idx = r if r < half_rows else r - half_rows
            col_offset = 0 if r < half_rows else 4
            color_hex = self.point_colors[col]           

            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            is_checked = (r < 3)
            chk.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
            chk.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, col_offset + 0, chk)

            if is_checked:
                self.selected_order.append(col)

            name_item = QTableWidgetItem(col)
            name_item.setData(Qt.UserRole, col)
            name_item.setToolTip("点击在左侧 Y 轴单独聚焦该测点刻度")
            self.table.setItem(row_idx, col_offset + 1, name_item)

            color_item = QTableWidgetItem(self._create_color_icon(color_hex), ' 选择')
            color_item.setTextAlignment(Qt.AlignCenter)
            color_item.setData(Qt.UserRole, col)
            color_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row_idx, col_offset + 2, color_item)

            v_min, v_max = np.nanmin(numeric_data[col]), np.nanmax(numeric_data[col])
            range_item = QTableWidgetItem(f'{v_min:.2f} ~ {v_max:.2f}')
            range_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, col_offset + 3, range_item)
        
        self.table.blockSignals(False)

        self.process_data_normalization()
        self.rebuild_waterfall_layout()
        self.apply_viewport_range()

    def process_data_normalization(self):
        self.data_ranges.clear()
        self.normalized_01_data.clear()

        for col, arr in self.raw_data.items():
            min_v = float(np.nanmin(arr))
            max_v = float(np.nanmax(arr))
            self.data_ranges[col] = (min_v, max_v)
            diff = max_v - min_v

            if diff <= 1e-6:
                self.normalized_01_data[col] = np.full_like(arr, 0.5, dtype=np.float32)
                continue

            is_zero_stop_type = (min_v >= -1e-4) and (min_v < 0.1) and (diff > 1.0)

            if is_zero_stop_type:
                threshold = max(0.01, diff * 0.005)
                non_zero_mask = arr > threshold
                res = np.zeros_like(arr, dtype=np.float32)

                if np.any(non_zero_mask):
                    nz_min = float(np.nanmin(arr[non_zero_mask]))
                    nz_diff = max_v - nz_min
                    if nz_diff > 1e-4:
                        res[non_zero_mask] = 0.12 + ((arr[non_zero_mask] - nz_min) / nz_diff) * 0.82
                    else:
                        res[non_zero_mask] = 0.50
                res[~non_zero_mask] = 0.03
                self.normalized_01_data[col] = res
            else:
                norm_linear = 0.08 + ((arr - min_v) / diff) * 0.84
                self.normalized_01_data[col] = norm_linear.astype(np.float32)

    def on_view_range_changed(self, view_box, range_box):
        if self.timestamps is None or len(self.timestamps) == 0:
            return
        self.lod_timer.start(50)

    def update_visible_curves_lod(self):
        if self.timestamps is None or len(self.timestamps) == 0 or len(self.curves) == 0:
            return

        vb = self.plot_widget.getViewBox()
        (vx_min, vx_max), _ = vb.viewRange()

        total_pts = len(self.timestamps)
        margin = (vx_max - vx_min) * 0.05
        left_val = vx_min - margin
        right_val = vx_max + margin

        i_start = np.searchsorted(self.timestamps, left_val, side='left')
        i_end = np.searchsorted(self.timestamps, right_val, side='right')

        i_start = max(0, i_start)
        i_end = min(total_pts, max(i_start + 2, i_end))
        visible_count = i_end - i_start

        target_pts = 3000
        need_downsample = visible_count > target_pts

        for col, curve in self.curves.items():
            if col not in self.waterfall_data:
                continue
            
            x_seg = self.timestamps[i_start:i_end]
            y_seg = self.waterfall_data[col][i_start:i_end]

            if need_downsample:
                num_buckets = target_pts // 2
                bucket_size = visible_count / num_buckets

                idx_list = []
                for b in range(num_buckets):
                    s = int(b * bucket_size)
                    e = int((b + 1) * bucket_size) if b < num_buckets - 1 else visible_count
                    if s >= e:
                        continue
                    sub_y = y_seg[s:e]
                    min_idx = s + int(np.argmin(sub_y))
                    max_idx = s + int(np.argmax(sub_y))
                    if min_idx < max_idx:
                        idx_list.extend([min_idx, max_idx])
                    else:
                        idx_list.extend([max_idx, min_idx])
                
                idx_arr = np.array(idx_list, dtype=np.int64)
                curve.setData(x=x_seg[idx_arr], y=y_seg[idx_arr])
            else:
                curve.setData(x=x_seg, y=y_seg)

    def apply_viewport_range(self):
        if self.timestamps is None or len(self.timestamps) == 0:
            return

        data_start = float(self.timestamps[0])
        data_end = float(self.timestamps[-1])
        total_duration = max(1.0, data_end - data_start)

        current_span_text = self.combo_span.currentText()
        span_sec = self.span_seconds_map.get(current_span_text, 600)

        vb = self.plot_widget.getViewBox()

        if span_sec is None:
            vb.setRange(xRange=(data_start, data_end), yRange=(0, 100), padding=0.01)
        else:
            picker_sec = float(self.dt_picker.dateTime().toSecsSinceEpoch())
            if total_duration <= span_sec:
                view_left = data_start
                view_right = data_end
            else:
                view_left = picker_sec
                if view_left < data_start:
                    view_left = data_start
                if view_left + span_sec > data_end:
                    view_left = max(data_start, data_end - span_sec)
                view_right = view_left + span_sec

            vb.setRange(xRange=(view_left, view_right), yRange=(0, 100), padding=0.01)

        self.update_visible_curves_lod()

    def on_plot_clicked(self, event):
        pos = event.scenePos()
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
        click_x = mouse_point.x()

        if self.timestamps is None or len(self.timestamps) == 0 or len(self.curves) == 0:
            return

        idx = int(np.clip(np.searchsorted(self.timestamps, click_x), 0, len(self.timestamps) - 1))
        real_time_sec = self.timestamps[idx]
        dt_str = datetime.fromtimestamp(real_time_sec).strftime('%Y-%m-%d %H:%M:%S')

        self.cursor_line.setPos(real_time_sec)
        self.cursor_line.setVisible(True)

        # 严格按照当前各泳道的实际空间位置（lane_top 降序）从上到下排列文字
        sorted_visible_cols = sorted(
            self.curves.keys(),
            key=lambda c: self.lane_ranges.get(c, (0, 0))[1], 
            reverse=True
        )

        lines = [f"<b style='color:#333;'>时间: {dt_str}</b>"]
        for col in sorted_visible_cols:
            if col in self.raw_data:
                val = self.raw_data[col][idx]
                color = self.point_colors.get(col, '#2383E2')
                lines.append(f"<span style='color:{color}; font-weight:bold;'>■ {col}: {val:.2f}</span>")

        self.cursor_text.setHtml("<br>".join(lines))
        vb = self.plot_widget.getViewBox()
        (vx_min, vx_max), _ = vb.viewRange()
        text_x = real_time_sec + (vx_max - vx_min) * 0.02 if (real_time_sec < vx_max - (vx_max - vx_min) * 0.3) else real_time_sec - (vx_max - vx_min) * 0.25
        self.cursor_text.setPos(text_x, 95)
        self.cursor_text.setVisible(True)

    def hide_cursor(self):
        self.cursor_line.setVisible(False)
        self.cursor_text.setVisible(False)

    def on_cell_clicked(self, row, col):
        if col in (2, 6):
            item = self.table.item(row, col)
            if not item:
                return
            col_name = item.data(Qt.UserRole)
            if not col_name:
                return

            initial_color = QColor(self.point_colors.get(col_name, '#2383E2'))
            chosen_color = QColorDialog.getColor(initial_color, self, f"选择 [{col_name}] 曲线颜色")
            if chosen_color.isValid():
                hex_color = chosen_color.name()
                self.point_colors[col_name] = hex_color
                item.setIcon(self._create_color_icon(hex_color))
                if col_name in self.curves:
                    pen = pg.mkPen(color=hex_color, width=1.5)
                    self.curves[col_name].setPen(pen)
                self.update_right_axis_scales()
            return

        if col in (1, 5):
            name_item = self.table.item(row, col)
            if not name_item:
                return
            col_name = name_item.data(Qt.UserRole)
            if col_name:
                self.focus_point_yaxis(col_name)

    def focus_point_yaxis(self, col_name: str):
        if col_name not in self.lane_ranges or col_name not in self.data_ranges:
            return

        self.active_focus_col = col_name
        lane_bottom, lane_top = self.lane_ranges[col_name]
        v_min, v_max = self.data_ranges[col_name]

        y_pos_min = lane_bottom + 0.08 * (lane_top - lane_bottom)
        y_pos_mid = lane_bottom + 0.50 * (lane_top - lane_bottom)
        y_pos_max = lane_bottom + 0.92 * (lane_top - lane_bottom)

        v_mid = (v_min + v_max) / 2.0

        ticks = [
            (y_pos_max, f'{v_max:.2f}'),
            (y_pos_mid, f'{v_mid:.2f}'),
            (y_pos_min, f'{v_min:.2f}')
        ]

        y_axis = self.plot_widget.getAxis('left')
        y_axis.setLabel(f'{col_name} 量程', color=self.point_colors.get(col_name, '#2F3437'))
        y_axis.setTicks([ticks])

    def update_right_axis_scales(self):
        configs = []
        # 按泳道空间从上到下自绘右侧专属彩色标尺
        sorted_visible_cols = sorted(
            self.curves.keys(),
            key=lambda c: self.lane_ranges.get(c, (0, 0))[1], 
            reverse=True
        )
        for col in sorted_visible_cols:
            if col in self.lane_ranges and col in self.data_ranges:
                lane_bottom, lane_top = self.lane_ranges[col]
                v_min, v_max = self.data_ranges[col]
                hex_color = self.point_colors.get(col, '#2383E2')
                configs.append((lane_bottom, lane_top, v_min, v_max, hex_color))
        self.right_axis.set_track_configs(configs)

    def rebuild_waterfall_layout(self):
        # 严格使用已记录的选择先后次序，过滤出有效测点
        visible_cols = [col for col in self.selected_order if col in self.normalized_01_data]

        # 清除既有分割线
        for line in self.lane_dividers:
            self.plot_widget.removeItem(line)
        self.lane_dividers.clear()
        self.lane_ranges.clear()

        # 移除未显示的曲线对象
        for col in list(self.curves.keys()):
            if col not in visible_cols:
                self.plot_widget.removeItem(self.curves[col])
                del self.curves[col]

        n = len(visible_cols)

        # 高度逻辑：<= 8 条维持 480px；> 8 条每增加一条动态追加 60px
        if n <= 8:
            dynamic_height = 480
        else:
            dynamic_height = 480 + (n - 8) * 60
        self.chart_card.setFixedHeight(dynamic_height)

        if n == 0 or self.timestamps is None or len(self.timestamps) == 0:
            self.waterfall_data.clear()
            self.plot_widget.getAxis('left').setTicks(None)
            self.plot_widget.getAxis('left').setLabel('')
            self.right_axis.set_track_configs([])
            self.hide_cursor()
            return

        # 1-8 条按实际勾选数量 n 均分 Y 轴量程（1条整屏、2条半屏、3条1/3屏...）
        lane_height = 96.0 / n
        gap = lane_height * 0.08 if n > 1 else 0.0

        # 自下而上分配：先勾选的（idx 较小）排在最底部泳道，后勾选的向上叠加
        for idx, col in enumerate(visible_cols):
            lane_bottom = 2.0 + idx * lane_height
            lane_top = lane_bottom + (lane_height - gap)
            self.lane_ranges[col] = (lane_bottom, lane_top)

            norm_arr = self.normalized_01_data[col]
            full_y = lane_bottom + norm_arr * (lane_top - lane_bottom)
            self.waterfall_data[col] = full_y

            # 绘制泳道间虚线分割线（在每层顶部绘制，最高层不画）
            if idx < n - 1:
                div_y = lane_top + gap / 2.0
                div_line = pg.InfiniteLine(pos=div_y, angle=0, pen=pg.mkPen('#787774', style=Qt.DashLine))
                self.plot_widget.addItem(div_line)
                self.lane_dividers.append(div_line)

            col_color = self.point_colors.get(col, '#2383E2')
            pen = pg.mkPen(color=col_color, width=1.5)

            if col not in self.curves:
                self.curves[col] = self.plot_widget.plot(pen=pen)
            else:
                self.curves[col].setPen(pen)

        self.update_right_axis_scales()
        self.update_visible_curves_lod()

        # 聚焦左侧主 Y 轴刻度
        if self.active_focus_col in visible_cols:
            self.focus_point_yaxis(self.active_focus_col)
        elif len(visible_cols) > 0:
            self.focus_point_yaxis(visible_cols[-1])

    def on_table_item_changed(self, item):
        col_idx = item.column()
        if col_idx in (0, 4):
            row = item.row()
            name_item = self.table.item(row, col_idx + 1)
            if not name_item:
                return
            col_name = name_item.data(Qt.UserRole)
            if not col_name:
                return

            # 根据勾选动作实时维护先进先出/先后次序
            if item.checkState() == Qt.Checked:
                if col_name not in self.selected_order:
                    self.selected_order.append(col_name)
            else:
                if col_name in self.selected_order:
                    self.selected_order.remove(col_name)

            self.rebuild_waterfall_layout()