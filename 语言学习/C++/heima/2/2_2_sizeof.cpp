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