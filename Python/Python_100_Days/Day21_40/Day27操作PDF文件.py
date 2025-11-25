# Day 27 操作PDF文件
# ==================== 导入模块 ====================
import os
import PyPDF2
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas 

# ==================== 主 程 序 ====================
# 从PDF中提取文本
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'res')

reader = PyPDF2.PdfReader(file_path + '/test.pdf')
# 通过PyPDF2.PdfReader()创建一个读取器对象reader
# 创建后reader就包含了该PDF文件所有信息
for page in reader.pages:
# reader.pages是一个包含 PDF 所有页面对象的列表
    print(page.extract_text())
    # 对每个页面对象page，调用extract_text()方法，该方法会提取当前页面中的文本内容

# 旋转和叠加页面
reader = PyPDF2.PdfReader(file_path + '/XGBoost.pdf') # 创建PDF读取器，用于读取原文件内容
writer = PyPDF2.PdfWriter() # 创建PDF写入器，用于构建新的PDF文件

for no, page in enumerate(reader.pages):
    if no % 2 == 0:
        new_page = page.rotate(-90) # 页面逆时针旋转90度
    else:
        new_page = page.rotate(90)  # 页面顺时针旋转90度 
    writer.add_page(new_page)       # 旋转后的页面写入新页面

with open(file_path + '/temp.pdf', 'wb') as file_obj: # 以二进制写入模式创建/打开temp.pdf文件
    writer.write(file_obj)               # 把写入器中所有处理后的页面写入新文件

# 加密PDF文件
reader = PyPDF2.PdfReader(file_path + '/XGBoost.pdf')
writer = PyPDF2.PdfWriter()

for page in reader.pages:
    writer.add_page(page)

writer.encrypt('foobared')

with open(file_path + '/temp_1.pdf', 'wb') as file_obj:
    writer.write(file_obj)

# 批量添加水印
reader1 = PyPDF2.PdfReader(file_path + '/XGBoost.pdf')
reader2 = PyPDF2.PdfReader(file_path + '/watermark.pdf')
writer = PyPDF2.PdfWriter()
watermark_page = reader2.pages[0]

for page in reader1.pages:
    page.merge_page(watermark_page)
    writer.add_page(page)

with open(file_path + '/temp_2.pdf', 'wb') as file_obj:
    writer.write(file_obj)

# 创建PDF文件略