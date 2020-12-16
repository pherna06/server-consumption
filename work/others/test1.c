#include <stdlib.h>

int main()
{
    int x = rand();
    while(1)
    {
        x += x * 3 % x;
    }

    return 0;
}