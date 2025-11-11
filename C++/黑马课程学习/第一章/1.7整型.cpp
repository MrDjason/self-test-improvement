#include<iostream>
using namespace std;
/*
区别在储存空间不同
short 短整型 至少2字节
int 整型 至少4字节
long 长整型 至少4字节
long long 至少8字节
*/
int main(){
    //整型
    //1.短整型
    short num1=10;
    //2.整型
    int num2=10;
    //3.长整型
    long num3 = 10;
    //4.长长整型
    long long num4 = 10;
    cout << "num1 = " << num1 << endl;
    cout << "num2 = " << num2 << endl;
    cout << "num3 = " << num3 << endl;
    cout << "num4 = " << num4 << endl;
    return 0;
}