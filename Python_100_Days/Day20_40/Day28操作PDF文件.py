# Day 28 操作PDF
# ==================== 导入模块 ====================
import PyPDF2
import os 
# ==================== 主 程 序 ====================
script_dir = os.path.

reader = PyPDF2.PdfReader('test.pdf')
for page in reader.pages:
    print(page.extract_text())