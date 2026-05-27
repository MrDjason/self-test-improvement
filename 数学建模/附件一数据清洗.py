import re
import pandas as pd
import os
from docx import Document

def extract_float(value_str):
    """从带后缀的字符串中提取数字，如 '42.55 ± 4.05 d' -> 42.55"""
    if not value_str or value_str.strip() in ('--', '—', '', 'nd'):
        return 0.0
    match = re.search(r'(\d+\.?\d*)', value_str)
    return float(match.group(1)) if match else 0.0

def is_header_row(row):
    """判断某行是否为表头行（包含 '0d' 或 '1d' 等天数）"""
    for cell in row:
        if re.match(r'\d+d', str(cell)):
            return True
    return False

def is_category_header(cell):
    """判断单元格是否为类别/种类等表头文字"""
    if not cell:
        return False
    return any(keyword in cell for keyword in ['类别', '种类', '化合物', '浓度'])

def parse_oil_data(rows, oil_name):
    """
    解析单个油品的所有行（已经切分好的从表头到下一个表头之间的行）
    rows: list of list，包含表头行和数据行
    """
    # 找到表头行（第一个包含天数的行）
    header_idx = None
    for i, row in enumerate(rows):
        if is_header_row(row):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"在 {oil_name} 的数据中未找到表头行")

    header_row = rows[header_idx]
    # 提取天数列索引
    time_indices = []
    day_mapping = {}
    for idx, cell in enumerate(header_row):
        if re.match(r'\d+d', str(cell)):
            day_num = re.search(r'(\d+)', str(cell)).group(1)
            col_name = f'day_{day_num}'
            time_indices.append(idx)
            day_mapping[idx] = col_name

    # 数据行从 header_idx+1 开始
    data_rows = rows[header_idx+1:]

    records = []
    current_category = None

    for row in data_rows:
        if len(row) < 2:
            continue

        # 跳过明显的小计/总和行
        if len(row) > 1 and ('小计' in str(row[1]) or '总和' in str(row[1]) or '合计' in str(row[1])):
            continue

        # 跳过表头说明行（如第一列为“类别”或“种类”等）
        if is_category_header(row[0]) and (len(row) > 1 and is_category_header(row[1])):
            continue

        # 类别更新：第一列非空且不是表头说明文字
        if row[0] and row[0].strip() != '' and not is_category_header(row[0]):
            current_category = row[0].strip()

        compound = row[1] if len(row) > 1 else ''
        if not compound or compound == '' or compound == '化合物':
            continue

        # 提取各天浓度
        conc = {}
        for idx in time_indices:
            if idx < len(row):
                val = extract_float(row[idx])
            else:
                val = 0.0
            conc[day_mapping[idx]] = val

        record = {
            'oil_type': oil_name,
            'category': current_category,
            'compound': compound,
            **conc
        }
        records.append(record)

    return pd.DataFrame(records)

def main():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '附件.docx')
    output_path = os.path.join(os.path.dirname(file_path), '核桃油VOCs清洗后数据.csv')

    doc = Document(file_path)

    # 收集所有表格的所有行
    all_rows = []
    for table in doc.tables:
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            all_rows.append(row_data)

    # 找出所有表头行的索引（行号）
    header_indices = [i for i, row in enumerate(all_rows) if is_header_row(row)]
    print(f"找到 {len(header_indices)} 个表头行，位置：{header_indices}")

    if len(header_indices) < 3:
        print("警告：表头行数量少于3，可能解析有误，将尝试按顺序分配油品名称")

    # 根据表头行切分数据，每组对应一个油品
    oil_names = ['香玲', '鲁光', '山核桃']
    dfs = []

    for idx, start in enumerate(header_indices):
        end = header_indices[idx+1] if idx+1 < len(header_indices) else len(all_rows)
        oil_rows = all_rows[start:end]

        # 尝试从数据中识别油品名称（查找前几行是否包含品种名）
        detected_name = None
        for row in oil_rows[:20]:
            row_str = ' '.join(str(cell) for cell in row)
            if '香玲' in row_str:
                detected_name = '香玲'
                break
            if '鲁光' in row_str:
                detected_name = '鲁光'
                break
            if '山核桃' in row_str:
                detected_name = '山核桃'
                break

        # 如果没找到，按顺序分配
        if detected_name is None and idx < len(oil_names):
            detected_name = oil_names[idx]
        elif detected_name is None:
            detected_name = f'未知{idx+1}'

        print(f"处理 {detected_name}，数据行范围 {start}~{end}")
        df = parse_oil_data(oil_rows, detected_name)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    # 调整列顺序
    first_cols = ['oil_type', 'category', 'compound']
    day_cols = [col for col in final_df.columns if col.startswith('day_')]
    day_cols_sorted = sorted(day_cols, key=lambda x: int(x.split('_')[1]))
    final_df = final_df[first_cols + day_cols_sorted]

    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存至：{output_path}")
    print(f"总行数：{len(final_df)}")
    print("前10行预览：")
    print(final_df.head(10))

if __name__ == '__main__':
    main()