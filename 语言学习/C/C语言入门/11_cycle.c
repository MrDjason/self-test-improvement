#include <stdio.h>

int main()
{
    for (int i = 1; i < 5 ; i++)
    {
        for (int j = 1; j <= 5; j++)
        {
            printf("内循环执行%d次\n", j);
            // break; // 跳出内循环
            goto a;
        }
        printf("内循环结束\n");
        printf("---------\n");
    }
    a:printf("外循环结束\n");
    return 0;
}
