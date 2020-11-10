import os
import signal

def process(core):
    command = "taskset -c " + str(core) + " "
    command += "/home/pherna06/venv-esfinge/server-consumption/work/test"

    os.system(command)

pid = []
for i in range(8):
    pid.append(os.fork())
    if pid[i] == 0: #child
        process(i)
        exit()

os.system('read -n 1 -s -r -p "Press any key to continue"')

for i in range(8):
    os.kill(pid[i], signal.SIGKILL)
