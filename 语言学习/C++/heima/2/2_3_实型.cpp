# include <iostream>
using namespace std;

int main()
{
    /*
    float 4
    double 8
    */
   // 默认情况下输出一个小数，会显示出6位有效数字
   float f1 = 3.14; // 默认double 3.1400000
   float f2 = 3.14f; // 加f则float
   cout << "f1 = " << f1 << endl;
   double d1 = 3.1415926;
   cout << "d1 = " << d1 << endl;

   cout << "float占用的内存空间为：" << sizeof(float) << endl;
   cout << "double占用的内存空间为：" << sizeof(double) << endl;
   
   // 科学计数法
   float f3 = 3e2;
   cout << "f3=" << f3 << endl;
   system("pause");
   return 0;
}