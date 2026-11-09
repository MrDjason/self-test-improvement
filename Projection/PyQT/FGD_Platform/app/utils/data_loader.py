from PySide6.QtCore import QThread, Signal
import pandas as pd
import numpy as np

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