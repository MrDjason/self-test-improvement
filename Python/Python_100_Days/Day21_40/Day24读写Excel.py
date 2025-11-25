# Day 24 读写Excel
# ==================== 基本信息 ====================
'''
Python 操作 Excel 需要三方库的支持，如果要兼容 Excel 2007 以前的版本，也就是xls格式的 Excel 文件，可以使用三方库xlrd和xlwt
前者用于读 Excel 文件，后者用于写 Excel 文件。

如果使用较新版本的 Excel，即xlsx格式的 Excel 文件，可以使用openpyxl库
'''
# ==================== 导入模块 ====================
import xlrd
import xlwt
import random
import os
# ==================== 主 程 序 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'res', '阿里巴巴2020年股票数据.xls')

wb = xlrd.open_workbook(file_path)
sheetnames = wb.sheet_names()
print(sheetnames)

sheet = wb.sheet_by_name(sheetnames[0])

print(sheet.nrows, sheet.ncols)

for row in range(sheet.nrows):
    for col in range(sheet.ncols):
        value = sheet.cell(row, col).value
        # 获取第row行第col列表格值
        if row > 0:
        # 判断当前是否不为第一行
            if col == 0:
            # 如果是第一列，处理日期信息
                value = xlrd.xldate_as_tuple(value, 0)
                # xlrd库中用于将 Excel 的 “日期数值” 转换为 Python 元组的函数
                # Excel中的日期本质是特殊数值。0代表从1900年开始的日期基准
                value = f'{value[0]}年{value[1]:>02d}月{value[2]:>02d}日'
            else:
            # 当前不为第一列
                value = f'{value:.2f}'
                # 数值保留两位小数
        print(value, end='\t')
    print()

last_cell_type = sheet.cell_type(sheet.nrows - 1,sheet.ncols -1)
# sheet.cell_type(rowx, colx) xlrd库中用于获取指定单元格数据类型的方法
# sheet.nrows - 1 获取总行最后一个表格的索引
print(last_cell_type)
# 0 - 空值，1 - 字符串，2 - 数字，3 - 日期，4 - 布尔，5 - 错误
print(sheet.row_values(0))
# 获取第零行指定列范围的数据
print(sheet.row_slice(3, 0, 5))
# 获取第4行第0列到第四列的数据
# 第一个参数代表行索引，第二个和第三个参数代表列的开始（含）和结束（不含）索引

# 写Excel
student_names = ['刘备', '关羽', '张飞', '马超', '赵云']
scores = [[random.randrange(50,101) for _ in range(3)] for _ in range(5)]
# 生成一个三列五行列表
wb = xlwt.Workbook()
# 创建工作表对象
sheet = wb.add_sheet('一年级二班')
# 添加表头数据
titles = ('姓名', '语文', '数学', '英语')
for index, title in enumerate(titles):
    sheet.write(0, index, title)
for row in range(len(scores)):
    sheet.write(row+1, 0, student_names[row])
    for col in range(len(scores[row])):
        sheet.write(row+1, col+1, scores[row][col])

file_path_1 = os.path.join(script_dir, 'res', '考试成绩表.xls')
wb.save(file_path_1)

# 调整单元格格式
# 还可以为单元格设置样式，主要包括字体（Font）、对齐方式（Alignment）、边框（Border）和背景（Background）的设置
header_style = xlwt.XFStyle()
# xlwt.XFStyle()：创建一个 “样式对象”，专门用于存储单元格样式的类
pattern = xlwt.Pattern()
# xlwt.Pattern()：创建一个 “图案对象”，专门用于设置单元格的背景图案和填充颜色
pattern.pattern = xlwt.Pattern.SOLID_PATTERN
# xlwt.Pattern.SOLID_PATTERN：xlwt 中定义的一个常量，表示 “实心填充” 模式
pattern.pattern_fore_colour = 5
# pattern_fore_colour：设置单元格背景填充色
header_style.pattern = pattern
# 把前面设置好的pattern对象关联到header_style样式对象中
titles = ('姓名', '语文', '数学', '英语')
for index, title in enumerate(titles):
    sheet.write(0, index, title, header_style)
font = xlwt.Font()

font.name = '华文楷体'
# 字体名称
font.height = 20 * 18
# 字体大小（20是基准单位，18表示18px）
font.bold = True
# 是否使用粗体
font.italic = False
# 是否使用斜体
font.colour_index = 1
# 字体颜色
header_style.font = font

align = xlwt.Alignment()
# 垂直方向的对齐方式
align.vert = xlwt.Alignment.VERT_CENTER
# 水平方向的对齐方式
align.horz = xlwt.Alignment.HORZ_CENTER
header_style.alignment = align

borders = xlwt.Borders()
props = (
    ('top', 'top_colour'), ('right', 'right_colour'),
    ('bottom', 'bottom_colour'), ('left', 'left_colour')
)
# 通过循环对四个方向的边框样式及颜色进行设定
for position, color in props:
    # 使用setattr内置函数动态给对象指定的属性赋值
    setattr(borders, position, xlwt.Borders.DASHED)
    setattr(borders, color, 5)
header_style.borders = borders

# 设置行高为40px
sheet.row(0).set_style(xlwt.easyxf(f'font:height {20 * 40}'))
titles = ('姓名', '语文', '数学', '英语')
for index, title in enumerate(titles):
    # 设置列宽为200px
    sheet.col(index).width = 20 * 200
    # 设置单元格的数据和样式
    sheet.write(0, index, title, header_style)