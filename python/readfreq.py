import cpufreq
import time


def main():
    cpu = cpufreq.cpuFreq()

    while(1):
        print(cpu.get_frequencies()[0])
        time.sleep(0.001)

if __name__ == '__main__':
    main()
