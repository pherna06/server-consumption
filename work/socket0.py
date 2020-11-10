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

print("Press Enter to continue")
os.system("read REPLY")

for i in range(8):
    print("Killing process in CPU.")
    os.kill(pid[i], signal.SIGKILL)
