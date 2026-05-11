#include <iostream>
#include <string> 
using namespace std;

int main(){
    // 1. C风格字符串
    // 注意1： char 字符串名[]
    // 注意2： 等号后面用双引号包含起来字符串
    char str[] = "hello world";
    cout << str <<endl;

    // C++ 风格字符串
    string str1 = "hello world";
    cout << str1 << endl;
    system("pause");
    return 0;
}