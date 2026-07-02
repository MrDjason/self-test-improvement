
#include <stdio.h>

int main()
{
    short a = 10;
    printf("%d\n", a);
    int b = 100; // 等价于 short in b = 100;
    printf("%d\n", b);
    long c = 1000L;
    printf("%ld\n", c);
    long long d = 10000LL;
    printf("%lld\n", d);

    // 利用 sizeof测量每一种数据类型占用字节
    printf("%zu\n", sizeof(short));
    printf("%zu\n", sizeof(a));
    printf("%zu\n", sizeof(b));
    printf("%zu\n", sizeof(c));
    printf("%zu\n", sizeof(d));
    return 0;
}