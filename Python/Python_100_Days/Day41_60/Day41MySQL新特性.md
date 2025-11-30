# MySQL新特性
### JSON
JSON 类型适合用于多选项 / 可扩展且非必选信息的存储场景，核心解决传统结构化存储的灵活性不足问题。  

- **JSON对象**
```json
{"name":"Jame", "tel":"132xxxxxxxx", "QQ":"123456"}
```
- **JSON数组**：
```json
[1, 2, 3]
[{"name":"kevin", "tel":"132xxxxxxxx"}, {"name":"jame", "tel":"132xxxxxxxx"}]
```
### 1.创建
```sql
CREATE TABLE `tb_test`
(
`user_id` bigint unsigned,
`login_info` json,
PRIMARY KEY(`user_id`)
);

INSERT INTO `tb_test`
VALUES
    (1, '{"tel": "13122335566", "QQ": "654321", "wechat": "jackfrued"}'),
    (2, '{"tel": "13599876543", "weibo": "wangdachui123"}');
```

### 2.查询
```sql
SELECT `user_id`
    , JSON_UNQUOTE(JSON_EXTRACT(`login_info`, '$.tel')) AS 手机号
    , JSON_UNQUOTE(JSON_EXTRACT(`login_info`, '$.wechat')) AS 微信
FROM `tb_test`;
-- 查询主键用户ID，唯一标识用户
-- JSON 类型的字段必须用专门的JSON提取函数JSON_EXTRACT()
-- JSON_EXTRACT(字段名, '$.key'):从login_info提取value
-- JSON_UNQUOTE():去掉双引号函数
```
- **更简便的写法**
```sql
SELECT `user_id`
    , `login_info` ->> '$.tel' AS 手机号
    , `login_info` ->> '$.wechat' AS 微信
FROM `tb_test`;
-- MySQL 5.7+ 支持 ->> 作为 JSON_EXTRACT + JSON_UNQUOTE 的合并简写
-- 字段名 ->> '$.key' = JSON_UNQUOTE(JSON_EXTRACT(字段名, '$.key'))
```

### 3.使用JSON类型保存用户画像数据
- **创建画像标签表**
```sql
CREATE TABLE `tb_tags`
(
`tag_id` int unsigned NOT NULL COMMENT '标签ID',
`tag_name` varchar(20) NOT NULL COMMENT '标签名',
PRIMARY KEY (`tag_id`)
);
-- 创建名为tag_dict的表，用于存储所有可选标签
-- 设tag_id为主键，确保每个标签ID唯一

INSERT INTO `tb_tags` (`tag_id`, `tag_name`)
VALUES
    (1, '70后'),
    (2, '80后'),
    (3, '90后'),
    (4, '00后'),
    (5, '爱运动'),
    (6, '高学历'),
    (7, '小资'),
    (8, '有房'),
    (9, '有车'),
    (10, '爱看电影'),
    (11, '爱网购'),
    (12, '常点外卖');
```
- **打标签**
```sql
CREATE TABLE `tb_users_tags`
(
`user_id`   bigint unsigned NOT NULL COMMENT '用户ID',
`user_tags` json            NOT NULL COMMENT '用户标签'
);   

INSERT INTO `tb_user_tags`
VALUES
    (1, '[2, 6, 8, 10]'),
    (2, '[3, 10 ,12]'),
    (3, '[3, 8, 9, 11]');
```

### JSON类型巧妙之处
```sql
-- user_tags -> '$'：JSON 路径表达式，$表示JSON 根节点，这里就是获取user_tags字段的整个 JSON 数组
-- 10 MEMBER OF (JSON数组) 判断单个值是否存在于JSON数组中
```

- **查询爱看电影的用户ID**
```sql
SELECT `user_id`
FROM `tb_users_tags`
WHERE 10 MEMBER OF (`user_tags` -> '$');
-- 判断10是否在user_tags的JSON数组中
```

- **查询爱看电影的80后**
```sql
SELECT `user_id`
FROM `tb_users_tags`
WHERE JSON_CONTAINS(`user_tags` -> '$', '[2, 10]');
-- JSON_CONTAINS(源JSON数组, 目标JSON数组)：判断源数组是否包含目标数组所有元素
```

- **查询爱看电影或80后或者90后用户ID**
```sql
SELECT `user_id`
FROM `tb_users_tags`
WHERE JSON_OVERLAPS(users_tags->'$', '[2,3,10]');
-- JSON_OVERLAPS(数组A, 数组B)：判断数组A和数组B是否有至少一个共同元素
```

### 窗口函数(OLAP联机分析和处理函数)
窗口函数核心价值是解决传统SQL（聚合函数、子查询、变量）难以高效、简洁实现的分析类需求——它既能实现分组/全局的精准计算，又能保留每条原始记录

```sql
<窗口函数> OVER (PARTITION BY <用于分组的列名> ORDER BY <用于排列的列名> ROWS BETWEEN ... AND ...)
<窗口函数> OVER (PARTITION BY <用于分组的列名> ORDER BY <用于排序的列名> RANGE BETWEEN ... AND ...)
-- 窗口函数可放置专用窗口函数：lead、lag、first_value、last_value、rank、dense_rank和row_number等
-- 聚合函数：sum、avg、max、min和count等
```
- **例子1：查询按月薪从高到低排在第4到第6名的员工的姓名和月薪**：
```sql
SELECT *
 FROM (SELECT `ename`
            , `sal`
            , ROW_NUMBER() over (ORDER BY `sal` DESC) AS `rk`
        FROM `tb_emp`) AS `temp`
WHERE `rk` between 4 and 6;
```
- **例子2：查询每个部门月薪最高的两名的员工的姓名和部门名称**：
```sql
SELECT ename, sal, dname FROM (
    SELECT ename, sal, dno,
           RANK() OVER (PARTITION BY dno ORDER BY sal DESC) AS rank  
           -- 按部门分组，组内月薪降序排名
    FROM tb_emp
) temp NATURAL JOIN tb_dept  -- 关联部门表获取部门名称
WHERE rank <= 2;  -- 过滤组内前2名
```