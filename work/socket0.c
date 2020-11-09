#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <sys/types.h>

int main()
{
    pid_t pid[8];

    for (int i = 0; i < 8; ++i)
    {
        pid[i] = fork();
        if (pid[i] == 0)
        {
            // Child.
            int e = execl(
                "/usr/bin/taskset",
                "taskset", "-c", i,
                "/home/pherna06/venv-esfinge/server-consumption/work/test");

            if (e == -1)
            {
                printf("execl error");
                return;
            }
            break;
        }        
    }

    printf("Press any key to kill processes.");
    int c = getchar();

    for (int i = 0; i < 8; ++i)
    {
        kill(pid[i], SIGKILL);
    }

    return 0;
}