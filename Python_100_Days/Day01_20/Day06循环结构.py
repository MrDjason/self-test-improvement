# Day 06 循环结构
import time
for _ in range(3): # 不需要循环变量用_代替
    print(_)
    # time.sleep(1)

print(sum(range(2,101,2))) # 100内偶数相加

sum = 0
for i in range(101):
    sum += i
print(sum)

sum=0
i=1
while i<=100:
    sum+=i
    i+=1
print(sum)

# 九九乘法表
for i in range(1,10):
    for j in range(1,i+1):
        print(f'{i} * {j} = {i * j}', end = '\t') # print()结束后会换行打印 end='\t'用制表符替换换行符
    print() # 换行

# 素数判断
num = int(input('请输入一个正整数判断是否为素数：'))
is_prime = True
for i in range(2,num):
    if num % i ==0:
        is_prime = False
        break
if is_prime:
    print(num, '这个数是素数')
else:
    print(num, '这个数不是素数')
