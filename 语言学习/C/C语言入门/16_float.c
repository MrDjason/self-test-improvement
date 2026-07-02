#include <stdio.h>

int main()
{
    float a = 3.14F;
    printf("%.2f\n", a);
    
    double b = 1.78;
    printf("%.2lf\n", b);

    long double c = 3.1415926L;
    printf("%.2Lf\n", c); // MinGW64不支持，输出0

    printf("%zu\n", sizeof(a));
    printf("%zu\n", sizeof(b));
    printf("%zu\n", sizeof(c));
    return 0;
}