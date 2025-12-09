# 视图、函数和过程

### 插入数据
```sql
-- 创建数据库hrs并指定默认字符集
create database `hrs` default charset utf8mb4

-- 切换到hrs数据库
use `hrs`;

-- 创建部门表
create table `tb_dept`
(
`dno` int not null comment '编号',
`dname` varchar(10) not null comment '名称',
`dloc` varchar(20) not null comment '所在地',
primary key (`dno`)
);

-- 插入4个部门
insert into `tb_dept` values
    (10, '会计部', '北京'),
    (20, '研发部', '成都'),
    (30, '销售部', '重庆'),
    (40, '运维部', '深圳');

-- 创建员工表
create table `tb_emp`
(
`eno` int not null comment '员工编号',
`ename` varchar(20) not null comment '员工姓名',
`job` varchar(20) not null comment '员工职位',
`mgr` int comment '主管编号',
`sal` int not null comment '员工月薪',
`comm` int comment '每月补贴',
`dno` int not null comment '所在部门编号',
primary key (`eno`),
constraint `fk_emp_mgr` foreign key (`mgr`) references tb_emp (`eno`),
constraint `fk_emp_dno` foreign key (`dno`) references tb_dept (`dno`)
);

-- 插入14个员工
insert into `tb_emp` values
    (7800, '张三丰', '总裁', null, 9000, 1200, 20),
    (2056, '乔峰', '分析师', 7800, 5000, 1500, 20),
    (3088, '李莫愁', '设计师', 2056, 3500, 800, 20),
    (3211, '张无忌', '程序员', 2056, 3200, null, 20),
    (3233, '丘处机', '程序员', 2056, 3400, null, 20),
    (3251, '张翠山', '程序员', 2056, 4000, null, 20),
    (5566, '宋远桥', '会计师', 7800, 4000, 1000, 10),
    (5234, '郭靖', '出纳', 5566, 2000, null, 10),
    (3344, '黄蓉', '销售主管', 7800, 3000, 800, 30),
    (1359, '胡一刀', '销售员', 3344, 1800, 200, 30),
    (4466, '苗人凤', '销售员', 3344, 2500, null, 30),
    (3244, '欧阳锋', '程序员', 3088, 3200, null, 20),
    (3577, '杨过', '会计', 5566, 2200, null, 10),
    (3588, '朱九真', '会计', 5566, 2500, null, 10);
```

### 视图
视图是虚拟的表，是一种虚拟结构，是保存好的SQL查询语句，查询视图会动态执行该SQL返回结果。
```sql
create view `vw_emp_simple`
as 
select `eno`,
       `ename`,
       `job`,
       `dno`,
from `tb_emp`;
-- 创建视图，只查询编号、姓名、工作、所属部门

-- 查询视图
select * from `vw_emp_simple`;

+------+-----------+--------------+-----+
| eno  | ename     | job          | dno |
+------+-----------+--------------+-----+
| 1359 | 胡二刀    | 销售员       |  30 |
| 2056 | 乔峰      | 分析师       |  20 |
| 3088 | 李莫愁    | 设计师       |  20 |
| 3211 | 张无忌    | 程序员       |  20 |
| 3233 | 丘处机    | 程序员       |  20 |
| 3244 | 欧阳锋    | 程序员       |  20 |
| 3251 | 张翠山    | 程序员       |  20 |
| 3344 | 黄蓉      | 销售主管     |  30 |
| 3577 | 杨过      | 会计         |  10 |
| 3588 | 朱九真    | 会计         |  10 |
| 4466 | 苗人凤    | 销售员       |  30 |
| 5234 | 郭靖      | 出纳         |  10 |
| 5566 | 宋远桥    | 会计师       |  10 |
| 7800 | 张三丰    | 总裁         |  20 |
+------+-----------+--------------+-----+

-- 删除视图
drop view if exists `vm_emp_simple`;
```

### 函数
函数用来封装功能上相对独立且会被重复使用的代码。下面例子通过自定义函数实现阶段超长字符串的功能。
```sql
delimiter $$ 
-- Mysql默认用;作为SQL语句的结束符，但函数内需要用到；
-- 为了避免Mysql提前判定语句结束，先把结束符临时改为$$
create function fn_truncate_string( -- 创建函数名
    content varchar(10000), -- 传入要处理的字符串
    max_length int unsigned -- 允许的最大长度
) returns varchar(10000) no sql -- 传入允许的最大长度并标记该函数不执行任何SQL操作
begin
    declare result varchar(10000) default content; -- 声明局部变量result，默认值是传入的content
    if char_length(content) > max_length then -- 判断：如果源字符串长度超过max_length
        set result = left(content, max_length); -- 取content前max_length个字符
        set result = concat(result, '......'); -- 拼接省略号
    end if;
    return result; -- 返回处理后的字符串
end $$
delimiter;

-- 使用自定义函数
select fn_truncate_string('和我在成都的街头走一走，直到所有的灯都熄灭了也不停留', 10) as short_string;

+--------------------------------------+
| short_string                         |
+--------------------------------------+
| 和我在成都的街头走一……                 |
+--------------------------------------+
```

### 过程
过程（储存过程）是事先编译好存储在数据库中的一组SQL集合，专为解决单条SQL无法完成的复杂业务场景设计。核心价值如下
1.**简化开发**：封装多步复杂操作，应用程序只需调用过程名，无需重复编写多条 SQL；
2.**提升性能**：减少应用与数据库的通信次数，且预编译特性比逐条执行 SQL 更快；
3.**保障数据一致性**：通过事务控制多步操作的原子性（要么全成、要么全败）；
4.**安全隔离**：不暴露数据表底层细节，降低数据泄露/误操作风险。
```sql
delimiter $$
create procedure sp_upgrade_salary() -- 创建一个储存过程名字
begin
    declare flag boolean default 1; -- 声明一个局部变量flag boolean布尔型 默认值 1

    declare continue handler for sqlexception set flag=0; -- 声明 继续型异常处理器（遇到异常不终止过程执行继续往下走）

    start transaction; -- 开启Mysql事务环境，保证后续update原子性执行
    update tb_emp set sal=sal+300 where dno=10;
    update tb_emp set sal=sal+800 where dno=20;
    update tb_emp set sal=sal+500 where dno=30;

    if flag then -- 根据flag的值决定提交还是回滚
        commit;
    else
        rollback;
    end if;
end $$
delimiter;

call sp_upgrade_salary(); -- 调用过程

drop procedure if exists sp_upgrade_salary; -- 删除过程
```