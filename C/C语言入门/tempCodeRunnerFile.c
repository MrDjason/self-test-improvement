#include <stdio.h>

int main()
{
    int arr[3][5] = 
    {
        {1,2,3,4,5},
        {11,22,33,44,55},
        {111,222,333,444,555}
    };

    int (*p)[5] = arr;
    printf("%p\n", arr + 1);

    for (int i = 0; i < 3; i++)
    {   
        // 遍历一维数组
        for (int j = 0; j < 5; j++)
        {
            print("%d", *(*p+j));
        }
        // 换行
        printf("\n");
        // 移动二维数组的指针，继续遍历下一个一维数组
        p++;
    }


    return 0;
}