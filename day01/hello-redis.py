from redis import Redis
redis_cli = Redis(host='localhost', port=6379, db=0)
redis_cli.set('mingzi', 'itheima')

name = redis_cli.get('mingzi')
print(name)
# b'itheima' b是Python李bytes类型的标识，这是一串二进制字节形式的数据

redis_cli.delete('mingzi') 