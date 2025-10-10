# Day 26 操作Word和PPT
from docx import Document
from docx.shared import Cm, Pt

from docx.document import Document as Doc

document = Document()
# 创建代表Word文档的Doc对象
document.add_heading('快快乐乐学python', 0)
# 添加大标题
p = document.add_paragraph('Python是一门非常流行的编程语言，它')
run = p.add_run('简单易学')
run.bold = True  # 加粗
run.italic = True  # 斜体
