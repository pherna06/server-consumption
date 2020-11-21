import argparse
import pyRAPL
import time
import subprocess
import os
import cpufreq
from  filelock import FileLock
import numpy as np

# MODIFY ACCORDING TO YOUR MACHINE CPU CONFIGURATION.
CPU_LIST = list(range(16))
SOCKET_LIST = [0, 1]
SOCKET_DICT = {
        0: CPU_LIST[0:8], 
        1: CPU_LIST[9:16]
        }

# DEFAULT RANDOM MATRICES SIZE.
MATRIX_SIZE = 1000
LIST_SIZE = MATRIX_SIZE*MATRIX_SIZE

# cpufreq variables.
AVAILABLE_FREQS = sorted(cpufreq.cpuFreq().available_frequencies)

## TIME TEST
TIME_CPU = 0
TIME_REPS = 10

# NUMBER GENERATION
MAX_INT = 1000000

# LOG FILEG PATH
LOG_PATH = "/home/pherna06/venv-esfinge/server-consumption/log/"

#####################
# UTILITY FUNCTIONS #
#####################

def closest_frequency(freq):
    """
        closest_frequency calculates the minimum available frequency 
        which is greater or equal than the given frequency. 
        If given frequency is below the minimum allowed frequency 
        an error is raised.

        :freq: given frequency
        :return: minimum frequency greater or equal than :freq:
    """
    
    # Minimum frequency allowed by CPU.
    minf = AVAILABLE_FREQS[0]

    # Check if correct.
    if freq < minf:
        print(f"ERROR: Specified frequency {freq} is below the minimum allowed frequency {minf}.")
        exit()

    # Get minimum frequency greater(=) than 'freq'.
    selfreq = AVAILABLE_FREQS[0]
    for af in AVAILABLE_FREQS:
        if freq <= selfreq:
            break
        selfreq = af

    return selfreq

def lower_frequency(freq, rg):
    """
        lower_frequency sets the cores in :rg: to :freq: frequency,
        supposed that their actual frequencies are greater than :freq:.

        :freq: new lower frequency
        :rg: CPU cores to modify
    """
    cpu = cpufreq.cpuFreq()
    cpu.set_min_frequencies(freq, rg)
    cpu.set_max_frequencies(freq, rg)
    cpu.set_frequencies(freq, rg)

def raise_frequency(freq, rg):
    """
        raise_frequency sets the cores in :rg: to :freq: frequency,
        supposed that their actual frequencyes are lower than :freq:.

        :freq: new greater frequency
        :rg: CPU cores to modify
    """
    cpu = cpufreq.cpuFreq()
    cpu.set_max_frequencies(freq, rg)
    cpu.set_min_frequencies(freq, rg)
    cpu.set_frequencies(freq, rg)

def process_fork(work, core, size, wpath, lock):
    """
        process_fork is used to handle forked processes. The
        affinity of the process is set to :core:. The operation
        to be executed and measured is given by :work:. The execution
        time is written in shared file :wpath:, concurrent-safe with :lock:.

        :work: string that specifies the operation to be handled
        :core: CPU core to set affinity
        :size: dimension of elements used in the operation
        :wpath: path of write file shared by forked processes
        :lock: lock for write file safe-concurrency
    """
    # Set core affinity.
    pid = os.getpid()
    command = "taskset -cp " + str(core) + " " + str(pid)
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)

    # Pick work operation:
    op = OPERATIONS[work]
    worktime = op(size)

    # Write in forks file.
    with FileLock(lock):
        wf = open(wpath, 'a')
        wf.write(f"{core} {worktime}\n")
        wf.close()

def produce_logs(workstr, results):
    """
        produce_logs writes the time results of :workstr: operation
        in the corresponding log files.
        A read-friendly .log file and a raw-data .csv are created.

        :workstr: str with name of operation, for log files naming.
        :results: a 2 dimension dictionary which stores execution time
                    in function of frequency and core.
    """
    # Log files paths.
    logpath = LOG_PATH + workstr + '.log'
    csvpath = LOG_PATH + workstr + '.csv'

    # Removing previous log files.
    if os.path.exists(logpath):
        os.remove(logpath)
    if os.path.exists(csvpath):
        os.remove(csvpath)

    # Writing log results.
    logf = open(logpath, 'w')
    csvf = open(csvpath, 'w')

    for freq in results:
        freqmhz = int(freq/1000)
        
        csvf.write(f"{freq}\n")
        logf.write(f"Frequency: {freqmhz} MHz\n")
        logf.write("-------------------------\n") # 25-
        logf.write("CPU   Time (s)\n")

        for core in sorted(results[freq]):
            csvf.write(f"{core}, {results[freq][core]}\n")
            logf.write(f"{core:<3}   {results[freq][core]:.3f}\n")

        logf.write("####################\n\n") # 25#


#########################################################

##############
# OPERATIONS #
##############

def int_product(size):
    """
        int_product performs a product of integer matrices and measures
        execution time. Matrices dimension is :size: x :size:.

        :size: dimension of square matrix
        :return: execution time of matrix product
    """
    # Random matrix generation.
    matA = np.random.randint(MAX_INT, size=(MATRIX_SIZE, MATRIX_SIZE))
    matB = np.random.randint(MAX_INT, size=(MATRIX_SIZE, MATRIX_SIZE))

    # TIME: operation.
    start = time.time()
    matC = np.matmul(matA, matB)
    end = time.time()

    return (end - start)

def float_product(size):
    """
        float_product performs a product of real matrices and measures
        execution time. Matrice dimension is :size: x :size:.

        :size: dimension of square matrix
        :return: execution time of matrix product
    """
    # Random matrix generation.
    matA = np.random.rand(size, size)
    matB = np.random.rand(size, size)

    # TIME: operation.
    start = time.time()
    matC = np.matmul(matA, matB)
    end = time.time()

    return (end - start)

OPERATIONS = {
        'intproduct': int_product,
        'floatproduct': float_product
        }

########################################################


def test_operation(work, freqs, rg, size):
    """
        test_operation sets the environment to take measures of 
        the operation specified in :work: parameter.
        For each frequency in :freqs:, a forked process is created
        for each core in :rg:, in which the operation is measured.
        Results are stored in a concurrent-safe file written by each
        process; then stored in the respective log files.

        :work: string that specifies the operation
        :freqs: frequencies in which the operation will be measured
        :rg: CPU cores in which the operation will be executed
        :size: dimension of elements in the operation
    """
    # Set frequencies to minimum allowed frequency.
    # So changing the frequency implies raising it.
    minf = AVAILABLE_FREQS[0]
    lower_frequency(minf, rg)

    # Dict to store results.
    results = {}

    # Forks write file and lock:
    temppath = "data.temp"
    templock = "data.temp.lock"
    
    # Measure exec time for each frequency in each implied core.
    freqs = sorted(freqs)
    for freq in freqs:
        raise_frequency(freq, rg)

        print(f"Testing with frequency of {int(freq/1000)} MHz.")
        
        # Remove conflicting file.
        if os.path.exists(temppath):
            os.remove(temppath)

        # Create file.
        wf = open(temppath, 'a+')
        wf.close()
        
        # Forks pid list.
        pidls = []
        for core in rg:
            pidls.append( os.fork() )
            if pidls[-1] == 0:
                process_fork(work, core, size, temppath, templock)
                exit(0)

        # Wait for all forked processes
        for pid in pidls:
            os.waitpid(pid, 0)

        # Get this frequency results.
        wf = open(temppath, 'r')
        wflines = wf.readlines()
        wf.close()
        
        results[freq] = {}
        for line in wflines:
            corestr, worktimestr = line.split()
            core = int(corestr)
            worktime = float(worktimestr)

            results[freq][core] = worktime

        os.remove(temppath)
        os.remove(templock)

    # Writing log files
    produce_logs(work, results)
    

############################################################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    socket_help = "sockets that will be measured.\n"
    socket_help += "all sockets selected by default"
    parser.add_argument('-s', '--socket', help=socket_help, nargs='+', type=int, default=SOCKET_LIST)

    freq_help =  "FREQ frequencies in which the selected operation will be tested.\n"
    freq_help += "all available frequencies set by default"
    parser.add_argument('-f', '--freq', help=freq_help, nargs='+', default=argparse.SUPPRESS)

    dim_help = "random matrices dimension: DIM x DIM.\n"
    dim_help += "set to {} by default".format(MATRIX_SIZE)
    parser.add_argument('-d', '--dim', help=dim_help, type=int, default=MATRIX_SIZE)

    work_help = "numpy action to be tested: intproduct transpose sort"
    parser.add_argument('work', help=work_help)

    intprod_help = "multiplication of random integer matrices" 
    trans_help = "transposal of random matrix"
    sort_help = "sorting of random list"

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    sockets = args.socket

    try:
        pyRAPL.setup(
            devices = [pyRAPL.Device.PKG],
            socket_ids = sockets
            )
    except:
        print("ERROR: check if selected sockets exist.")
        exit()

    # Cores implied
    rg = []
    [ rg.extend(SOCKET_DICT[skt]) for skt in sockets ]

    # Gets closest frequencies to the selected ones.
    freqs = []
    if 'freq' in args:
        userfs = args.freq
        for freq in userfs:
            freqs.append( closest_frequency(freq * 1000 ) )
    else:
        freqs = AVAILABLE_FREQS

    size = args.dim
    
    if args.work == 'intproduct':
        test_operation('intproduct', freqs, rg, size)
    if args.work == 'floatproduct':
        test_operation('floatproduct', freqs, rg, size)
    if args.work == 'sort':
        pass


if __name__ == '__main__':
    main()
