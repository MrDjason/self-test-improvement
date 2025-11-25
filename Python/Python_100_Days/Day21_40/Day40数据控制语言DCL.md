# DCL
### 定义
数据库服务器通常包含了非常重要的数据，可以通过访问控制来确保这些数据的安全，而DCL就是解决这一问题的。它可以为指定的用户授予访问权限或者从指定用户处召回指定的权限。

### 创建用户
```sql
-- 创建用户wangdachui，访问口令为Wang.618
CREATE USER 'wangdachui'@'%' IDENTIFIED BY 'Wang.618';
-- 该用户可以从任意主机访问数据库服务器

DROP USER IF EXISTS 'wangdachui'@'%';
-- wangdachui 只能从192.168.0.x 这个网段的主机访问数据库服务器
CREATE USER 'wangdachui'@'192.168.0.%' IDENTIFIED BY 'Wang.618'
```

### 授予权限
```sql
-- 为wangdachui授予查询库Database表table_name的权限
GRANT SELECT ON `Database_name`.`table_name` TO 'wangdachui'@'192.168.0.%';

-- 让wangdachui对Database库的所有对象都具有查询权限
GRANT SELECT ON `Database`.* TO 'wangdachui'@'192.168.0.%';

-- 希望wangdachui还有insert、delete、update权限
GRANT INSERT,DELETE,UPDATE ON `Database`.* TO 'wangdachui'@'192.168.0.%';

-- 授予wangdachui执行DDL的权限
GRANT CREATE, DROP, ALTER ON `Database`.* TO 'wangdachui'@'192.168.0.%';

-- wangdachui对所有数据库的所有对象都具备所有的操作权限
GRANT ALL PRIVILEGES ON *.* TO 'wangdachui'@'192.168.0.%';
```

### 召回权限
```sql
-- 召回wangdachui对Database的insert、delete、update权限
REVOKE INSERT, DELETE, UPDATE ON `Database`.* FROM 'wangdachui'@'192.168.0.%'
-- 召回所有权限
REVOKE ALL PRIVILEGES ON *.* FROM 'wangdachui'@'192.168.0.%';

FLUSH PRIVILEGES; -- 即时生效
```