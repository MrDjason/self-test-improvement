# Day 05 分支结构
# 赋值表达式格式化 同时输出「变量名」和「变量值」
bmi = 22.5
print(f'bmi = {bmi:.1f}')  # 输出：bmi = 22.2
print(f'{bmi = :.1f}')

# match和case构造分支结构
status_code = int(input("请输入错误代码："))
match status_code:
    case 400: description = 'Bad Request'
    case 401: description = 'Unauthorized'
print('状态码描述：', description)
