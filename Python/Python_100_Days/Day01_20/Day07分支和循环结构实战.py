# Day 07 分支和循环结构实战
# 100以内素数
prime_list =[]
for i in range(2,101):
    is_prime = True
    for j in range(2,i):
        if i%j == 0:
            is_prime = False
            break
    if is_prime:
        prime_list.append(i)

print(prime_list)

# 斐波那契数列 前20个数
Fibonacci_list = []
a,b =1,1
for _ in range(20):
    Fibonacci_list.append(a)
    a,b = b,a+b
print(Fibonacci_list)

# 水仙花数 寻找100-999中所有水仙花数 如153 = 1³ + 5³ + 3³,N位非负整数，其各位数字的 N次方和刚好等于该数本身
Narcissistic_list=[]
for num in range(100,1000):
    i = num//100 # 取百位
    j = num//10%10 # 153//10=15，15%10=5 取十位
    k = num%10 # 153%10=3 取个位
    if i**3 + j**3 + k**3 == num:
        Narcissistic_list.append(num)
print(Narcissistic_list)

# 百钱百鸡问题 公鸡 5 元一只，母鸡 3 元一只，小鸡 1 元三只，用 100 块钱买一百只鸡，问公鸡、母鸡、小鸡各有多少只？
# 5x + 3y + (z/3) = 100/z为3的倍数/公鸡0-20只，母鸡0-33只，小鸡0-100只
for i in range (21):
    for j in range(34):
        k = 100 - i - j
        if k>=0 and k%3==0 and 5*i+3*j+k//3==100:
            print(f'公鸡{i}只，母鸡{j}只，小鸡{k}只')