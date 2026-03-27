#include <stdio.h>
#define _CRT_SECURE_NO_WARNINGS

int add(int num1, int num2);
int minus(int num1, int num2);
int mult(int num1, int num2);
int div(int num1, int num2);

int main()
{
    /*
    定义加、减、乘、除、四个函数
    用户键盘录入三个数字
    前两个表示参与计算的数字
    第三个数字表示调用的函数
    1:加法
    2:减法
    3:乘法
    4:除法
    */

    // 定义一个数组去装四个函数的指针
    int (*arr[4])(int, int) = {add, minus, mult, div};
    // 用户录入数据
    int num1;
    int num2;
    printf("请输入两个数字参与计算\n");
    scanf("%d%d", &num1, &num2);
    printf("%d\n", num1);
    printf("%d\n", num2);

    int choose;
    printf("请录入数字表示要进行的计算\n");
    scanf("%d", &choose);
    // 调用不同函数
    int res = (arr[choose - 1](num1, num2));
    printf("%d\n", res);
    return 0;
}

int add(int num1, int num2)
{
    return num1 + num2;
}

int minus(int num1, int num2)
{
    return num1 - num2;
}

int mult(int num1, int num2)
{
    return num1 * num2;
}

int div(int num1, int num2)
{
    return num1 / num2;
}