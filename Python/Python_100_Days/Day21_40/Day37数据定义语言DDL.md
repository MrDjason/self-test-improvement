# DDL
### 定义
DDL为数据定义语言，主要用于创建、删除、修改数据库中的对象，比如创建、删除和修改二维表，核心关键字包括`create`、`drop`和`alter`
### 功能
（1）建库建表（2）删除表（3）修改表
### 任务
实现一个简单的选课系统数据库，将数据库命名为`school`

### 1.1 建库
```sql
-- 创建数据库
CREATE DATABASE `school`;

-- 创建数据库并指定默认字符集
CREATE DATABASE `school`
DEFAULT CHARACTER SET utf8mb4; -- 支持所有UTF-8字符（含Emoji、特殊符号）

--创建数据库并纸锭默认字符集和排序规则（最规范）
CREATE DATABASE `school`
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_general_ci; -- 不区分大小写排序
```

### 1.2 使用库
```sql
USE `school`; -- 切换后，所有SQL默认操作school库
```

### 1.3 删除库
```sql
-- 删除名为 school 的整个数据库（含所有表和数据）
DROP DATABASE `school`;
-- 若存在则删除，不存在则不报错
DROP DATABASE IF EXISTS `school`; 
```

### 1.4 修改库
```sql
-- 重命名库，需要新创建库并迁移旧库数据至新库

-- 调整数据库的默认字符集和排序规则
ALTER DATABASE `school`
DEFAULT CHARACTER SET `utf8mb4` -- 新字符集
COLLATE utf8mb4_general_ci; -- 新排序规则
```

### 2.1 建表
- **主键**：给表中每条记录分配 “唯一身份标识”，强制非空、重复自动创建索引、加速查询，其他表可以通过外键引用该主键
```sql
-- 基础建表
CREATE TABLE `table_name` (
    `id` INT,
    `name` VARCHAR(50),
    `age` TINYINT,
    `gender` CHAR(1),
    `create_time` DATETIME
);

-- 规范建表（含约束：主键、非空、自增、默认值）
CREATE TABLE `table_name` (
-- 安全建表
-- CREATE TABLE IF NOT EXISTS `table_name` (
    `id` INT PRIMARY KEY AUTO_INCREMENT, -- 主键、自增
    `name` VARCHAR(50) NOT NULL, -- 非空
    `age` TINYINT UNSIGNED DEFAULT 18, -- 无符号（非负整数） + 默认值18 
    `gender` CHAR(1) CHECK (gender IN ('男', '女', '未知')), --约束性别只能是三种
    `phone` VARCHAR(20) UNIQUE, -- 手机号唯一
    `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP -- 默认值：当前系统时间
);
```

### 2.2 使用表
- **核心操作**：查/增/改/删数据
```sql
-- 查询表
SELECT * FROM `table_name`; -- 查询表中所有数据
SELECT `name`, `age`, `gender` FROM `student` WHERE `age` > 20; -- 筛选条件查询

-- 插入表数据
INSERT INTO `table_name` (`name`) VALUES('张三'); -- 指定字段插入数据

-- 更新表数据
UPDATE `table_name` SET `age` = 23 WHERE `id` = 1;

-- 删除表数据
DELETE FROM `table_name` WHERE `id` = 1; -- 只删除id=1的记录
DELETE FROM `table_name`; -- 删除表中所有数据
```

### 2.3 删除表
```sql
DROP TABLE `table_name`; -- 直接删除表（表不存在则报错）
DROP TABLE IF EXISTS `table_name`;
```

### 2.4 修改表
```sql
-- 修改表名
ALTER TABLE `table_name` RENAME TO `new_table_name`;

-- 给表添加新字段
ALTER TABLE `table_name` ADD COLUMN `address` VARCHAR(200) DEFAULT '未填写';

-- 修改已有字段
ALTER TABLE `table_name` MODIFY COLUMN `age` INT UNSIGNED; -- 将age从tinyint改成int

-- 修改字段名+字段类型
ALTER TABLE `table_name` CHANGE COLUMN `phone` `tel` VARCHAR(20) UNIQUE;

-- 删除表字段
ALTER TABLE `table_name` DROP COLUMN `address`;

-- 修改字段默认值
ALTER TABLE `table_name` ALTER COLUMN `gender` SET DEFAULT '未知';

-- 给字段添加/删除非空约束
ALTER TABLE `table_name` MODIFY COLUMN `name` VARCHAR(50) NULL; -- NULL空，NOT NULL非空
```

### 完整程序
```sql
-- 如果存在名为school的数据库就删除它
DROP DATABASE IF EXISTS `school`;

-- 创建名为school的数据库并设置默认的字符集和排序方式
CREATE DATABASE `school` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

-- 创建学院表
CREATE TABLE `tb_college`
(
`col_id`    int unsigned AUTO_INCREMENT      COMMENT '编号',
`col_name`  varchar(50)  NOT NULL            COMMENT '名称',
`col_intro` varchar(500) NOT NULL DEFAULT '' COMMENT '介绍',
PRIMARY KEY (`col_id`)
);

-- 创建学生表
CREATE TABLE `tb_student`
(
`stu_id`    int unsigned NOT NULL           COMMENT '学号',
`stu_name`  varchar(20)  NOT NULL           COMMENT '姓名',
`stu_sex`   boolean      NOT NULL DEFAULT 1 COMMENT '性别',
`stu_birth` date         NOT NULL           COMMENT '出生日期',
`stu_addr`  varchar(255) DEFAULT ''         COMMENT '贯籍',
`col_id`    int unsigned NOT NULL           COMMENT '所属学院',
PRIMARY KEY (`stu_id`),
CONSTRAINT `fk_student_col_id` FOREIGN KEY (`col_id`) REFERENCES `tb_college`(`col_id`),
);

-- 创建教师表
CREATE TABLE `tb_teacher`
(
`tea_id`    int unsigned NOT NULL                COMMENT '工号',
`tea_name`  varchar(20)  NOT NULL                COMMENT '姓名',
`tea_title` varchar(10)  NOT NULL DEFAULT '助教' COMMENT '职称',
`col_id`    int unsigned NOT NULL                COMMENT '所属学院',
PRIMARY KEY (`tea_id`),
CONSTRAINT `fk_teacher_col_id` FOREIGN KEY (`col_id`) REFERENCES `tb_college` (`col_id`)
);

-- 创建课程表
CREATE TABLE `tb_course`
(
`cou_id`     int unsigned NOT NULL COMMENT '编号',
`cou_name`   varchar(50)  NOT NULL COMMENT '名称',
`cou_credit` int          NOT NULL COMMENT '学分',
`tea_id`     int unsigned NOT NULL COMMENT '授课老师',
PRIMARY KEY (`cou_id`),
CONSTRAINT `fk_course_tea_id` FOREIGN KEY (`tea_id`) REFERENCES `tb_teacher` (`tea_id`)
);

-- 创建选课记录表
CREATE TABLE `tb_record`
(
`rec_id`   bigint unsigned AUTO_INCREMENT COMMENT '选课记录号',
`stu_id`   int unsigned    NOT NULL       COMMENT '学号',
`cou_id`   int unsigned    NOT NULL       COMMENT '课程编号',
`sel_date` date            NOT NULL       COMMENT '选课日期',
`score`    decimal(4,1)                   COMMENT '考试成绩',
PRIMARY KEY (`rec_id`),
CONSTRAINT `fk_record_stu_id` FOREIGN KEY (`stu_id`) REFERENCES `tb_student` (`stu_id`),
CONSTRAINT `fk_record_cou_id` FOREIGN KEY (`cou_id`) REFERENCES `tb_course` (`cou_id`),
CONSTRAINT `uk_record_stu_cou` UNIQUE (`stu_id`, `cou_id`)
);
```