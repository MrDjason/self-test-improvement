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