# 索引
索引是关系型数据库提升查询性能的核心手段，类比书籍目录：以牺牲少量存储空间为代价，大幅减少查询耗时（避免全表扫描）
```sql
explain select * from tb_student where stuname='林震南'\G
*************************** 1. row ***************************  
           id: 1  
  select_type: SIMPLE  
        table: tb_student  
   partitions: NULL  
         type: ALL  
possible_keys: NULL  
          key: NULL  
      key_len: NULL  
          ref: NULL  
         rows: 11  
     filtered: 10.00  
        Extra: Using where  
1 row in set, 1 warning (0.00 sec)  
```


在学生姓名对应的列上创建索引，通过索引来加速查询。
```sql
create index idx_student_name on tb_student(stuname);
explain select * from tb_student where stuname='林震南'\G

*************************** 1. row ***************************
           id: 1  
  select_type: SIMPLE  
        table: tb_student
   partitions: NULL
         type: ref
possible_keys: idx_student_name
          key: idx_student_name
      key_len: 62
          ref: const
         rows: 1
     filtered: 100.00
        Extra: NULL
1 row in set, 1 warning (0.00 sec)
```
1.`type`从`ALL`变成`ref/eq_ref`  
2.`key`从`NULL`变成创建的索引名  
3.`rows`大幅减少  
索引生效、性能提升  
MySQL 中还允许创建前缀索引，即对索引字段的前N个字符创建索引，这样的话可以减少索引占用的空间（但节省了空间很有可能会浪费时间，时间和空间是不可调和的矛盾）


```sql
create index idx_student_name_1 on tb_student(stuname(1)); -- 根据学生姓名第一个字创建索引
explain select * from tb_student where stuname='林震南'\G

*************************** 1. row ***************************
           id: 1
  select_type: SIMPLE
        table: tb_student
   partitions: NULL
         type: ref
possible_keys: idx_student_name
          key: idx_student_name
      key_len: 5
          ref: const
         rows: 2
     filtered: 100.00
        Extra: Using where
1 row in set, 1 warning (0.00 sec)

alter table tb_student drop index idx_student_name; -- 删除索引
drop index idx_student_name on tb_student; -- 删除索引
```
rows变成了2行，因为表中有两个学生，用姓名第一个字作为索引，在查询时通过索引找到这两行  
1.最适合索引的列是出现在WHERE子句和连接子句中的列。  
2.索引列的基数越大（取值多、重复值少），索引的效果就越好。  
3.使用前缀索引可以减少索引占用的空间，内存中可以缓存更多的索引。  
4.索引不是越多越好，虽然索引加速了读操作（查询），但是写操作（增、删、改）都会变得更慢，因为数据的变化会导致索引的更新，就如同书籍章节的增删需要更新目录一样。  
5.使用 InnoDB 存储引擎时，表的普通索引都会保存主键的值，所以主键要尽可能选择较短的数据类型，这样可以有效的减少索引占用的空间，提升索引的缓存效果。