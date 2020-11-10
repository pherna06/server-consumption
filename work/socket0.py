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

c = input("Press any key to kill processes.")

for i in range(8):
    os.kill(pid[i], signal.SIGKILL)