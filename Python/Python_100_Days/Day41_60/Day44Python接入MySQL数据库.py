# Day 44 Python接入Mysql数据库
# ==================== 导入模块 ====================
import pymysql
# ==================== 导入模块 ====================
no = int(input('部门编号：'))
name = input('部门名称：')
location = input('部门所在地：')

# 创建连接
conn = pymysql.connect(host='127.0.0.1', port=3306,
                       user='guest', password='Guest.618',
                       database='hrs', charset ='utf8mb4')

try:
    # 获取游标对象
    with conn.cursor() as cursor:
        # 通过游标对象向数据库服务器发出SQL语句
        affected_rows = cursor.execute(
            'insert into `tb_dept` values (%s, %s, %s)',
            (no, name, location)
        )
        if affected_rows == 1:
            print('新增部门成功！')
    # 提交事务
    conn.commit()
except pymysql.MySQLError as err:
    # 回滚
    conn.rollback()
    print(type(err), err)
finally:
    # 关闭连接释放资源
    conn.close()
    

'''
创建guest并授权
create user 'guest'@'%' identified by 'Guest.618';
grant insert, delete, update, select on `hrs`.* to 'guest'@'%';
'''