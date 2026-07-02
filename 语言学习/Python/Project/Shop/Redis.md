# Redis
- **Redis**：一个高性能(运行在内存中)的key-value数据库(nosql)
- **Redis简介**  
1.Redis是一个开源使用ANSI C语言写、支持网络、可基于内存亦可持久化的日志型、Key-Value数据库，并提供多种语言API。  
2.Redis是NoSQL技术阵营中的一员，通过多种键值数据类型来适应不同场景下的存储需求  

- **学习目标**：    
1.能描绘出什么是nosql  
2.能说出redis的特点  
3.能写出redis中string类型数据的增删改查操作命令  
4.能够写出redis中hash类数据的增删改查相关命令  
5.能够说出redis中list保存的数据类型  
6.能够使用strictredis对象对string类型数据进行增删改查  

## NoSQL （Not Only SQL）
- **定义**：  
1.泛指非关系型数据库  
2.不支持SQL语法  
3.nosql存储的数据都是以key-value形式  
4.nosql没有通用语言  
5.适用于简单的数据查询场景

## Redis 语法
**基本语法**：`redis-cli`，连接本地的redis服务  ，使用PING检测redis服务是否启动  
```redis
$ redis-cli
redis 127.0.0.1:6379>
redis 127.0.0.1:6379> PING
PONG
```
# Redis 键与值

## Redis 键(Key)
Redis 中所有数据均以键值对（Key-Value）形式存储，Key 是访问数据的唯一标识
**语法**：  
Redis键命令的基本语法如下：
```bash
COMMAND KEY_NAME
```

## Redis 值(value)
Value有五种类型：  
1.字符串`string`  
2.哈希`hash`  
3.列表`list`  
4.集合`set`  
5.有序集合`zset`  

## 键值对数据操作行为
增删改查  

### 保存
如果设置的键不存在则为添加，如果设置的键已经存在则修改
- **设置键值**  
```
set key value
```
- **键值及过期时间，以秒为单位**  
```
setex key seconds value
```
- **设置多个键值**  
```
mset key1 value1 key2 value2
```

### 获取
- **获取**  
根据键值获取值，如果不存在此键则返回`nil`
```
get key
```
- **根据多个键获取多个值**
```
mget key1 key2
```

### 删除
删除键时会将值删除

## 键命令
- **查找键，参数支持正则表达式**
```
keys pattern
```
- **查看所有键**
```
keys *
```
- **判断键是否存在，存在返回`1`，不存在返回`0`**  
```
exists key1
```
- **设置过期时间，以秒为单位。如果没有指定过期时间则一直存在，直到使用`DEL`移除**  
```
expire key seconds
```
- **查看有效时间，以秒为单位**
```
ttl key
```

## Hash哈希
Redis 的 Hash（哈希）是一种嵌套式的复合键值对数据类型，外层有一个全局的 Key，作为整个哈希结构的标识
内层包含多个(字段:值)的键值对
- 因为是`hash`命令，所有指令前加`h`
- `hash`用于存储对象，对象的结构为属性、值
- **值**的类型为`string`

### 增加、修改
- **设置单个属性**  
```
hset key field value  
```
- **设置多个属性**  
```
hmset key field1 value1 field2 value2
```

### 获取
- **获取指定键所有的属性**
```
hkeys key
```
- **获取一个属性的值**
```
hget key field
```
- **获取多个属性的值**
```
hmget key field1 field2
```
- **获取所有属性的值**
```
hvals key
```

### 删除
- 删除整个hash键及值，使用`del`命令
- 删除属性，属性对应的值会被一起删除
```
hdel key field1 field2
```

## List列表
- 因为是`list`命令，所有指令前加`l`
- 列表的元素类型为`string`
- 按照插入顺序排序

### 增加
- 在左侧插入数据
```
lpush key value1 value2
```
- 在右侧插入数据
```
rpush key value1 value2
```

### 获取
- 返回列表里指定范围内的元素
  - `start`、`stop`为元素的下标索引
  - 索引从左侧开始，第一个元素为0
  - 索引可以是负数，表示从尾部开始计数，如`-1`表示最后一个元素
```
lrange key start stop
```

### 删除
- **删除指定元素**
  - 将列表中前`count`次出现的值为`value`的元素移除
  - `count>0`：从头往尾移除
  - `count<0`：从尾往头移除
  - `count=0`：移除所有
```
lrem key count value
```

## Set集合
- 无序集合
- 元素为string型
- 元素具有唯一性，不重复
- 说明：对于集合没有修改操作

### 增加
- 添加元素
```
sadd key member1 member2
```
- 返回所有元素
```
smembers key
```

### 删除
```
srem key
```

## Zset有序列表
- `sorted set` 有序集合
- 元素为`string`类型
- 元素具有唯一性，不重复
- 每个元素都会关联一个`double`类型的`score`，表示权重，通过权重将元素从小到大排序
- 说明：没有修改操作

### 增加
```
zadd key score1 member1 score2 member2...
```

### 获取
```
zrange key start stop
```

### 删除
```
zrem key member1 member2
```