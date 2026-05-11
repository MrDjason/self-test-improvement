# C++基础

## 1.C++初识

### 1.1 HelloWorld

```cpp
#include <iostream>
using namespace std;

int main()
{
    cout << "hello world" << endl;
    system("pause");
    return 0;
}
```

### 1.2 注释

```cpp
#include <iostream>
using namespace std;

int main()
{
    // 单行注释
    /* 
       多
       行
       注
       释
    */
    system("pause");
    return 0;
}
```

`//`：单行注释  
`/**/`：多行注释  

### 1.3 变量

```cpp
#include <iostream>
using namespace std;

int main()
{
    int a = 10;
    cout << "a = " << a << endl;
    system("pause");
    return 0;
}
```

### 1.4 常量

```cpp
#include <iostream>
using namespace std;
// #define 宏常量
// const 修饰该变量为常量不能修改
#define day 7
int main()
{
    cout << "一周一共有" << day << "天" << endl;
    const int month = 12;
    system("pause");
    return 0;
}
```
`#define`：预处理直接文本替换，只能通过`#define NUM 100`定义  
- 错误写法：
```cpp
#define NUM = 100
```
`const`：初始化后禁止重新赋值  
- 错误写法：
```cpp
const int a = 6;
a = 88;
```
`#define`：宏常量

### 1.5 关键字

关键字是C++系统预先定义、具有特定功能和含义的标识符，不能用作变量名、函数名、宏名等自定义名称

### 1.6 标识符命名规则

标识符是程序员自定义的名称，用于命名变量、函数、宏、数组等，必须遵循C++统一规则。
- 由字母、数字、下划线组成，不能包含空格、@、#、$等特殊符号
- 不能以数字开头
- 不能与C++关键字重名
- 区分大小写

## 2.数据类型

### 2.1 整型

```cpp
/*
short 短整型 2
int 整型 4
long 长整型 4或64位linux 8
longlong 长长整型 8
*/
#include <iostream>
using namespace std;

int main(){
    short num1 = 10;
    int num2 = 10;
    cout << "num1 = " << num1 << endl;
    cout << "num2 = " << num2 << endl;
    system("pause");
    return 0;
```

**常用整型及大小**

| 类型 | 关键字 | 通常大小（字节） | 取值范围（2的次方表示） |
| :--- | :--- | :--- | :--- |
| 短整型 | `short` | 2 | $-2^{15} \sim 2^{15}-1 \quad (\text{即 } -32768 \sim 32767)$ |
| 整型 | `int` | 4 | $-2^{31} \sim 2^{31}-1 \quad (\text{即 } -2,147,483,648 \sim 2,147,483,647)$ |
| 长整型（Windows） | `long` | 4 | $-2^{31} \sim 2^{31}-1 \quad (\text{同 } int)$ |
| 长整型（Linux 64位） | `long` | 8 | $-2^{63} \sim 2^{63}-1 \quad (\text{同 } long\ long)$ |
| 长长整型 | `long long` | 8 | $-2^{63} \sim 2^{63}-1 \quad (\text{约 } -9.22\times10^{18} \sim 9.22\times10^{18})$ |

**无符号整型**  
在不使用负数的情况下可以使用 `unsigned` 前缀，可使正数范围扩大一倍。

**确定大小**  
```cpp
cout << sizeof(short) << endl;   // 通常输出 2
cout << sizeof(int)   << endl;   // 通常输出 4
```

**整型溢出**

当赋予的值超出类型范围时，会发生溢出，导致数据截断或循环  

```cpp
short num = 32767;   // 最大值
num = num + 1;       // 结果变成 -32768（溢出回绕）
```

**整型字面量写法**

5. 整型字面量写法
- 十进制：
```cpp
int a = 42;
```

- 八进制（前缀 0）：
```cpp
int b = 052; // 十进制 42
```

- 十六进制（前缀 0x 或 0X）：
```cpp
int c = 0x2A; // 十进制 42
```

- 后缀指定类型：

  - `10L` → `long` 
  - `10LL` → `long long`
  - `10U` → `unsigned int`

### 2.2 sizeof

```cpp
#include <iostream>
using namespace std;

int main(){
    short num1 = 10;
    cout << "short占用空间:" << sizeof(num1) << endl;
    int num2 = 10;
    cout << "int占用空间" << sizeof(num2) << endl;
    long num3 =10;
    cout << "long占用空间" << sizeof(num3) << endl;
    long long num4 =10;
    cout << "long long占用空间" << sizeof(num4) << endl;
    system("pause");
    return 0;
}
```

### 2.3 实型

实型（浮点型）用于表示带小数点的数字

|类型|关键字|占用字节|有效数字位数（十进制）|说明|
|:---|:---|:---|:---|:---|
|单精度|`float`|4|6~7位|精度较低，节省内存|
|双精度|`double`|8|15~16位|默认浮点字面量类型，精度高|

**字面量写法**

```cpp
float f1 = 3.14;      // 3.14 默认是 double 类型，赋值给 float 会有隐式转换
float f2 = 3.14f;     // 加 f / F 后缀表示 float 字面量
double d1 = 3.14;     // 直接使用 double
double d2 = 3.14;     // 不加后缀默认就是 double
```

```cpp
#include <iostream>
using namespace std;

int main() {
    /*
    float  : 4 字节
    double : 8 字节
    */
    // 默认输出：6 位有效数字
    float f1 = 3.14;          // 字面量 double → 转 float
    float f2 = 3.14f;         // 明确使用 float 字面量
    cout << "f1 = " << f1 << endl;   // f1 = 3.14
    double d1 = 3.1415926;
    cout << "d1 = " << d1 << endl;   // d1 = 3.14159 (6位有效数字)
    return 0;
}
```

**科学计数法**

```cpp
float f2 = 3e2; // 3 * 10 ^ 2
```

### 2.4 字符型
- **作用**：字符型变量用于显示单个字符
- **语法**：`char ch = 'a';`
- 注意1：在显示字符型变量时，用单引号将字符括起来，不要使用双引号  
- 注意2：单引号内只能有一个字符，不可以是字符串  

> C和C++中字符变量只占用1个字节  
> 字符型变量并不是把字符本身放到内存中存储，而是将对应的ASCII编码放入到存储单元

```cpp
#include <iostream>
using namespace std;

int main(){
    // 1. 字符型变量创建方式
    char ch = 'a';
    cout << ch << endl; // a

    // 2. 字符型变量所占内存大小
    cout << sizeof(ch) << endl; // 1

    // 3. 字符型变量常见错误
    // char ch1 = "b"; 创建字符型变量要用单引号
    // char ch2 = 'abcdef' 创建字符型变量时候，单引号内只能有一个字符

    // 4. 字符型变量对应ASCII编码
    cout << (int)ch << endl; // 97

    system("pause");
    return 0;
}
```

### 2.5 转义字符
- **作用**：用于表示一些不能显示出来的ASCII字符
- 现阶段常用转义字符有：`\n \\ \t`

|转义字符|含义|ASCII码值（十进制）|
|:---:|:---|:---:|
|`\a`|警报（响铃）|007|
|`\b`|退格(BS)，将当前位置移到前一列|008|
|`\f`|换页(FF)，将当前位置移到下页开头|012|
|`\n`|换行(LF)，将当前位置移到下一行开头|010|
|`\r`|回车(CR)，将当前位置移到本行开头|013|
|`\t`|水平制表(HT)（跳到下一个TAB位置）|009|
|`\v`|垂直制表(VT)|011|
|`\\`|代表一个反斜线字符`\`|092|
|`\'`|代表一个单引号（撇号）字符|039|
|`\"`|代表一个双引号字符|034|
|`\?`|代表一个问号|063|

```cpp
#include <iostream>
using namespace std;

int main(){
    // 转义字符

    // 换行符 \n

    cout << "hello world\n" << endl;

    // 反斜杠 \\ 注释行末尾 \ = 续行符
    cout << "hello \\ world" << endl;

    // 水平制表符 \t
    cout << "hello \t world" << endl;

    system("pause");
    return 0;
}
```

- `行尾 \`：行末尾用反斜杠 等于 续行符

### 2.6 字符串型
- **作用**：用于表示一串字符
- **两种风格**：
  - 1.**C风格字符串**：`char 变量名[] = "字符串值"`
  - 2.**C++风格字符串**：`string 变量名 = "字符串值"`
    > 使用C++风格字符串，需要包含头文件 #include <string>

### 2.7 布尔类型 bool
- **作用**：布尔数据类型代表真或假的值
bool类型只有两个值：
- true 真 1
- false 假 0

```cpp
#include <iostream>
using namespace std;

int main(){
    // 1. 创建bool数据类型
    bool flag = true;
    cout << flag << endl; // 1

    flag = false;
    cout << flag << endl; // 0
    // 2. 查看bool类型所占内存空间
    cout << "布尔类型所占空间：" << sizeof(bool) << endl; // 1

    return 0;
}
```

### 2.8 数据的输入
- **作用**：用于从键盘获取数据
- **关键字**：cin >> 变量

实例：

```cpp
#include <iostream>
#include <string>
using namespace std;

int main(){
    // 1. 整型
    int a = 0;
    cout << a << endl;
    cout << "请给整型变量a赋值：" << endl;
    cin >> a;
    cout << "整型变量a是多少：" << a << endl;

    // 2. 浮点型
    float f = 3.14f;
    cout << "请给浮点型变量f赋值：" << endl;
    cin >> f;
    cout << "浮点型变量f是多少：" << f << endl;

    // 3. 字符型
    char ch = 'a';
    cout << "请给字符型变量ch赋值：" << endl;
    cin >> ch;
    cout << "字符型变量ch是什么：" << endl;

    // 4. 字符串型
    string str = "hello";
    cout << "请给字符串型变量str赋值：" << endl;
    cin >> str;
    cout << "字符串型变量str是什么：" << str << endl;

    // 5. 布尔类型
    bool flag = false;
    cout << "请给布尔类型flag赋值" << endl;
    cin >> flag;
    cout << "布尔类型flag是什么：" << flag << endl;
}
```

## 3.运算符
- **作用**：用于执行代码的运算

主要讲解以下几类运算符：

|运算符类型|作用|
|:---:|:---|
|算术运算符|用于处理四则运算|
|赋值运算符|用于将表达式的值赋给变量|
|比较运算符|用于表达式的比较，并返回一个真值或假值|
|逻辑运算符|用于根据表达式的值返回真值或假值|

### 3.1 算术运算符

|运算符|术语|示例|结果|
|:---:|:---:|:---:|:---:|
|`+`|正号|`+3`|`3`|
|`-`|负号|`-3`|`-3`|
|`+`|加|`10+5`|`15`|
|`-`|减|`10-5`|`5`|
|`*`|乘|`10*5`|`50`|
|`/`|除|`10/5`|`2`|
|`%`|取模(取余)|`10%3`|`1`|
|`++`|前置递增|`a=2;b=++a;`|`a=3;b=3;`|
|`++`|后置递增|`a=2;b=a++;`|`a=3;b=2;`|
|`--`|前置递减|`a=2;b=--a;`|`a=1;b=1;`|
|`--`|后置递减|`a=2;b=a--;`|`a=1;b=2;`|

- 加减乘除：
  - （1）除数不为零
  - （2）两个整数相除结果是整数
  - （3）两个小数无法做取模运算
- 递增递减：
  - （1）`++a`：前置递增，让变量a+1
  - （2）`b++`：后置递增，让变量b+1
  - （3）`前置递增`：先加后运算
  ```cpp
    int a = 10;
    int b = ++a2 * 10;
    cout << "a = " << endl; // 11
    cout << "b = " << endl; // 110
  ```
  - （4）`后置递增`：先运算后加
  ```cpp
    int a = 10;
    int b = a2++ * 10;
    cout << "a = " << endl; // 11
    cout << "b = " << endl; // 100
  ```

### 3.2 赋值运算符
- **作用**：用于将表达式的值赋给变量  
赋值运算符包括如下符号：  

|运算符|术语|示例|结果|
|:---:|:---:|:---:|:---:|
|=|赋值|a=2;b=3;|a=2;b=3;|
|+=|加等于|a=0;a+=2;|a=2;|
|-=|减等于|a=5;a-=3;|a=2;|
|*=|乘等于|a=2;a*=2;|a=4;|
|/=|除等于|a=4;a/=2;|a=2;|
|%=|模等于|a=3;a%2;|a=1;|

### 3.3 比较运算符
- **作用**：用于表达式的比较，并返回一个真值或假值  
比较运算符有以下符号：  

|运算符|术语|示例|结果|
|:---:|:---:|:---:|:---:|
|==|相等于|4==3|0|
|!=|不等于|4!=3|1|
|<|小于|4<3|0|
|>|大于|4>3|1|
|<=|小于等于|4<=3|0|
|>=|大于等于|4>=1|1|

例如：

```cpp
cout << (a == b) << endl;
```

### 3.4 逻辑运算符
- **作用**：用于根据表达式的值返回真值或假值
逻辑运算符有以下符号：

|运算符|术语|示例|结果|
|:---:|:---:|:---:|:---:|
|!|非|!a|如果a为假，则!a为真；如果a为真，则!a为假。|
|&&|与|a&&b|如果a和b都为真，则结果为真，否则为假。|
|\|\||或|a\|\|b|如果a和b有一个为真，则结果为真，二者都为假时，结果为假。|

示例：  
```cpp
int main(){
    int a = 10;
    // 逻辑非运算
    cout << !a << endl; // 0
    cout << !!a << endl; // 1

    // 逻辑与运算
    int b = 10;
    int c = 10;
    cout << (b && c) << endl; // 1
    c = 0;
    cout << (b && c) << endl; // 0

    // 逻辑或运算符
    // 略

    system("pause");
    return 0;
}
```

## 4.程序流程结构
C/C++支持最基本的三种程序运行结构：顺序结构、选择结构、循环结构  
- 顺序结构：程序按顺序执行，不发生跳转
- 选择结构：依据条件是否满足，有选择的执行相应功能
- 循环结构：依据条件是否满足，循环多次执行某段代码
- 
### 4.1 选择结构

#### 4.1.1 if语句
**作用**：执行满足条件语句

示例  
```cpp
#include <iostream>
using namespace std;

int main(){
    float condition = 0.0f;

    cout << "请输入数字判断" << endl;
    cin >> condition;
    if (condition >= 1) // if
    {
        cout << "condition >= 1" << endl;
        if (condition >= 100) // 嵌套if
        {
            cout << "condition >= 100" << endl; 
        }
    }
    else if ((0 <= condition)&&(condition < 1)) // else if
    {
        cout << "0 =<condition < 1" << endl;
    }
    else // else
    {
        cout << "condition < 0" << endl;
    }
    system("pause");
    return 0;
}
```

#### 4.1.2 三目运算符  
**作用**：通过三目运算符实现简单的判断  
**语法**：```表达式1 ? 表达式2 : 表达式3```  
**解释**：  
如果表达式1的值为真，执行表达式2，并返回表达式2的结果
如果表达式2的值为假，执行表达式3，并返回表达式3的结果

```cpp
#include <iostream>
using namespace std;

int main(){
    // 三目运算符
    // ab哪个变量大 赋值给c
    int a = 10;
    int b = 20;
    int c = 0;
    cout << a << " " << b << " " << c << endl; // 10 20 0
    c = a > b ? a : b; 
    cout << a << " " << b << " " << c << endl;// 10 20 20


    (a > b ? a : b) = 100;
    cout << a << " " << b << endl; // 10 100

    return 0;
}
```

三目运算符返回的是一个变量，可以继续进行操作

#### 4.1.3 Switch语句
**作用**：执行多条件分支语句
**语法**

```cpp
switch(表达式)
{   
    case 结果1: 执行语句; break;
    case 结果2: 执行语句; break;

    ...

    default:执行语句;break;
}
```

示例  
```cpp
#include <iostream>
using namespace std;

int main(){
    int score = 0;
    cout << "请输入分数：" << endl;
    cin >> score;

    switch (score)
    {
        case 60:
            cout << "及格" << endl;
            break; // 否则会出现穿透
        case 75:
            cout << "良好" << endl;
            // 输入75测试break的作用
        case 90:
            cout << "优秀" << endl;
        default:
            cout << "你的分数不是60、75、90 不予评价" << endl;
    }
    return 0;
}
```

`switch`优点：结构清晰、执行效率高
`switch`缺点：判断的时候只能是整型或者字符型，不可以是一个区间
> 注意：case中没有break那么程序会一直向下执行

### 4.2 循环结构

#### 4.2.1 while循环语句
**作用**：满足循环条件，执行循环语句
**语法**：```while (循环条件) {循环语句}```

```cpp
#include <iostream>
using namespace std;

int main(){
    int num = 0;
    while (num<10)
    {
        cout << num << endl;
        num++;
    }
    return 0;
}
```

#### 4.2.2 do...while
**作用**：满足循环条件，执行循环语句  
**语法**：```do {循环语句} while (循环条件)```  

```cpp
#include <iostream>
using namespace std;

int main()
{
    int num = 0;
    do
    {
        cout << num << endl;
        num ++;
    }
    while(num < 10);
    return 0;
}
```

与 while 先判断条件不同，do while 会先执行一次循环语句再判断循环条件

#### 4.2.3 for循环语句
**作用**：满足循环条件，执行循环语句
**语法**：```for(起始表达式;条件表达式;末尾循环体) {循环语句}```

示例  
```cpp
#include <iostream>
using namespace std;

int main()
{
    for (int i = 0; i < 10; i++ )
    {
        cout << i << endl;
    }
    return 0;
}
```

#### 4.2.4 嵌套循环
**作用**：在循环体中再嵌套一层循环，解决一些实际问题

### 4.3 跳转语句

#### 4.3.1 break语句
**作用**：用于跳出选择结构或者循环结构
break使用的时机：
- 出现再switch条件语句中，作用是终止case并跳出switch 
- 出现在循环语句中，作用是跳出当前的循环语句
- 出现在嵌套循环中，跳出最近的内层循环语句

#### 4.3.2 continue语句
**作用**：在循环语句中，跳过本次循环中余下未执行语句，继续执行下一次循环

#### 4.3.3 goto语句
**作用**：可以无条件跳转语句  
**语法**：```goto 标记```  
**解释**：如果标记的名称存在，执行到goto语句时，会跳转到标记的位置  

示例
```cpp
int main()
{
    cout << "a" << endl;
    goto FLAG;
    cout << "2" << endl;
    
    FLAG:
    cout << "5" << endl;
    return 0;
}
```

## 5.数组

### 5.1 概述
所谓数组，就是一个集合，里面存放了相同类型的数据元素
**特点1**：数组中的每个数据元素都是相同的数据类型
**特点2**：数组是由连续的内存位置组成的

### 5.2 一维数组

#### 5.2.1 一维数组的定义方式
一维数组定义的三种方式：  
1.```数据类型 数组名[数组长度];```   
2.```数据类型 数组名[数组长度] = {值1, 值2, ...}```   
3.```数据类型 数组名[] = {值1, 值2, ...}```   

```cpp
#include <iostream>
using namespace std;

int main()
{
    // 数据类型 数组名[数组长度]
    int arr[5];
    arr[0] = 10;
    arr[1] = 10;

    // 数据类型 数组名[数组长度] = {值1, 值2, ...}
    // 如果再初始化数据的时候，没有全部填写完，会用0来填补剩余数据
    int arr1[5] = {1, 2, 3, 4, 5};

    // 数据类型 数组名[] = {值1, 值2, ...}
    int arr2[] = {1,2,3,4,5,6,7,8,9};
    for(int i = 0; i < 10; i++)
    {
        cout << arr[i] << endl;
        cout << arr1[i] << endl;
        cout << arr2[i] << endl;
    }
    return 0;
}
```

#### 5.2.2 一维数组数组名
一维数组名称的用途：
1.可以统计整个数组在内存中的长度
2.可以获取数组在内存中的首地址

示例
```cpp
#include <iostream>
using namespace std;

int main()
{
    // 数组名用途
    // 1.可以获取整个数组占用内存空间大小
    int arr[10] = {1,2,3,4,5,6,7,8,9,10};
    cout << "整个数组所占内存空间为：" << sizeof(arr) << endl;
    cout << "每个元素所占内存空间为:" << sizeof(arr[0]) << endl;
    cout << "数组的元素个数为：" << sizeof(arr) / sizeof(arr[0]) << endl;

    // 2.可以通过数组名获取到数组首地址
    cout << "数组首地址为：" << (long long)arr << endl;
    cout << "数组中第一个元素地址为：" << (long long)&arr[0] << endl;
    cout << "数组中第二个元素地址为：" << (long long)&arr[1] << endl;
    return 0;
}
```

### 5.3.2 二维数组数组名
- 查看二维数组所占内存空间
- 获取二维数组首地址

示例  

```cpp
#include <iostream>
using namespace std;

int main()
{
    int arr[2][3] = 
    {
        {1,2,3},
        {4,5,6}
    };
    cout << "二维数组大小：" << sizeof(arr) << endl;
    cout << "二维数组一行大小：" << sizeof(arr[0]) << endl;
    cout << "二维数组元素大小：" << sizeof(arr[0][0]) << endl;
    cout << "二维数组行数为：" << sizeof(arr) / sizeof(arr[0]) << endl;
    cout << "二维数组列数为：" << sizeof(arr[0]) / sizeof(arr[0][0]) << endl;
    
    cout << "二维数组首地址为：" << (long long)arr << endl;
    cout << "二维数组第一行首地址为：" << (long long)arr[0] << endl;
    cout << "二维数组第二行首地址为：" << (long long)arr[1] << endl;

    cout << "二维数组第一个元素首地址：" << (long long)&arr[0][0] << endl;
    cout << "二维数组第二个元素首地址：" << (long long)&arr[0][1] << endl;
    return 0;
}
```

## 6.函数

### 6.1 概述
**作用**：讲一段经常使用的代码封装起来，减少重复代码

### 6.2 函数的定义
函数的定义一般主要有5个步骤：  
1、返回值类型  
2、函数名  
3、参数列表  
4、函数体语句  
5、return表达式  

```cpp
返回类型 函数名 (参数列表)
{
    函数体语句
    return 表达式 
}
```

示例  

```cpp
int add(int num1, int num2)
{
    int sum = num1 + num2;
    return sum;
}
```

### 6.3 函数的调用
**功能**：使用定义好的函数
**语法**：```函数名 (参数)```

示例  

```cpp
#include <iostream>
using namespace std;


int add(int num1, int num2)
{
    int sum = num1 + num2;
    return sum;
}

int main()
{
    int a = 10;
    int b = 20;
    int c = add(a, b);
    cout << c << endl;
    return 0;
}
```

### 6.4 值传递
- 所谓值传递，就是函数调用时实参将数值传入给形参
- 值传递时，如果形参发生，并不会影响实参

示例  

```cpp
#include <iostream>
using namespace std;

void swap(int num1, int num2)
{
    cout << "交换前：" << endl;
    cout << "num1 = " << num1 << endl; // 10
    cout << "num2 = " << num2 << endl; // 20

    int temp = num1;
    num1 = num2;
    num2 = temp;

    cout << "交换后：" << endl;
    cout << "num1 = " << num1 << endl; // 20
    cout << "num2 = " << num2 << endl; // 10
    // return; 返回值不需要可以不写return
}

int main()
{
    int a = 10;
    int b = 20;
    cout << "a = " << a << endl; // 10
    cout << "b = " << b << endl; // 20
    swap(a, b);
    cout << "a = " << a << endl; // 10
    cout << "b = " << b << endl; // 20

    return 0;
}
```

### 6.5 函数的常见样式
常见的函数样式有4种  
1.无参无返  
2.有参无返  
3.无参有返  
4.有参有返  

示例  

```cpp
#include <iostream>
using namespace std;
// 1.无参无返
void test01()
{
    cout << "无参无返" << endl;
}
// 2.有参无返
void test02(int a)
{
    cout << "有参无返 a = " << a << endl;
}
// 3.无参有返
int test03()
{
    cout << "无参有返" << endl;
    return 1000;
}

// 4.有参有返
int test04(int a)
{
    cout << "有参有返" << endl;
    return a;
}

int main()
{
    test01(); // 无参无返调用

    test02(100); //有参无返调用

    int num1 = test03(); // 无参有返调用
    cout << "num1 = " << num1 << endl;

    int num2 = test04(10000); // 有参有返函数调用
    cout << "num2 = " << num2 << endl;
    return 0;
}
```
### 6.6 函数的声明
**作用**：告诉编译器函数名称以及如何调用函数。函数的实际主体可以单独定义  

- 函数的声明可以多次，但是函数的定义只能有一次

示例  

```cpp
#include <iostream>
using namespace std;
// 声明可以多次，定义只能一次

// 函数声明
int max(int a, int b);
int max(int a, int b);

// 函数定义
int max(int a, int b)
{
    return a > b ? a : b;
}

int main()
{
    int a = 10;
    int b = 20;

    cout << max(a,b) << endl;
    return 0;
}
```

### 6.7 函数的分文件编写
**作用**：让代码结构更加清晰  
函数分文件编写一般有4个步骤  
1.创建后缀名为.h的头文件  
2.创建后缀名为.cpp的源文件  
3.在头文件中写函数的声明  
4.在源文件中写函数的定义  

示例  

```cpp
// swap.h文件
#include<iostream>
using namespace std;

//实现两个数字交换的函数声明
void swap(int a, int b)
```

```cpp
// swap.cpp文件

```


## 7.指针

## 8.结构体
