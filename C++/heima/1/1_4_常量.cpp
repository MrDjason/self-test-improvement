#include <iostream>
using namespace std;
// #define 宏常量
// const 修饰该变量为常量不能修改
#define day 7 // # define day = 14 则错误 day是常量一修改则报错

int main()
{

    cout << "一周一共有" << day << "天" << endl;

    // const修饰的变量也称为常量，一修改则报错

    const int month = 12;

    system("pause");
    return 0;
}