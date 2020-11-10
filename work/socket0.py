import os
import signal
import random

def process(core):
    x = random.randomint(1, 10000)
    while True:
        x += x * 3 % x

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
