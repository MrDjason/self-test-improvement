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
### 创建
```sql
CREATE TABLE `table_name`
(
`user_id` bigint unsigned,
`login_info` json,
PRIMARY KEY(`user_id`)
);

INSERT INTO `table_name`
VALUES
    (1, '{"tel": "13122335566", "QQ": "654321", "wechat": "jackfrued"}'),
    (2, '{"tel": "13599876543", "weibo": "wangdachui123"}');
```

### 查询
```sql
SELECT `user_id`
    , JSON_UNQUOTE(JSON_EXTRACT(`login_info`, '$.tel')) AS 手机号
    , JSON_UNQUOTE(JSON_EXTRACT(`login_info`, '$.wechat')) AS 微信
FROM `table_name`;
```
更简便的写法  
```sql
SELECT `user_id`
    , `login_info` ->> '$.tel' AS 手机号
    , `login_info` ->> `$.wechat` AS 微信
    FROM `table_name`;
```

### 使用JSON类型保存用户画像数据
- **创建画像标签表**
```sql
CREATE TABLE `table_name`
(
`tag_id` int unsigned NOT NULL COMMENT '标签ID',
`tag_name` varchar(20) NOT NULL COMMENT '标签名',
PRIMARY KET (`tag_id`)
);

INSERT INTO `table_name` (`tag_id`, `tag_name`)
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
CREATE TABLE `table_name`
(
`user_id`   bigint unsigned NOT NULL COMMENT '用户ID',
`user_tags` json            NOT NULL COMMENT '用户标签'
);   
``    
);
INSERT INTO `table_name`
VALUES
    (1, '[2, 6, 8, 10]'),
    (2, '[3, 10 ,12]'),
    (3, '[3, 8, 9, 11]');
```

### JSON类型巧妙之处
- **查询爱看电影的用户ID**
```sql
SELECT `user_id`
FROM `tb_users_tags`
WHERE 10 MEMBER OF (`user_tags` -> '$');
```

- **查询爱看电影的80后**
```sql
SELECT `user_id`
FROM `tb_users_tags`
WHERE JSON_CONTAINS(`user_tags` -> '$', '[2, 10]');
```

- **查询爱看电影的80后或者90后用户ID**
```sql
SELECT `user_id`
FROM `tb_users_tags`
WHERE JSON_OVERLAPS(users_tags->'$', '[2,3,10]');
```

### 窗口函数