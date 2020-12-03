import argparse
import pyRAPL
import time
import subprocess
import os
import signal
import cpufreq
from  filelock import FileLock
import numpy as np

# MODIFY ACCORDING TO YOUR MACHINE CPU CONFIGURATION.
CPU_LIST = list(range(16))
SOCKET_LIST = [0, 1]
SOCKET_DICT = {
        0: CPU_LIST[0:8], 
        1: CPU_LIST[8:16]
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

# ENERGY MEASURE
MEASURE_TIME = 2

# MAX FREQUENCY
NOMINAL_MAXFREQ = 2601000
REAL_MAXFREQ = 3000000

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

def get_cores(sockets):
    """
        get_cores retrieves the cores assigned to the socket ids
        in :sockets:

        :sockets: list of socket ids
        :return: list of affected cores
    """
    rg = []
    for skt in sockets:
        rg.extend(SOCKET_DICT[skt])

    return rg


def time_fork(work, core, size, wpath, lock):
    """
        time_fork is used to handle forked processes for execution
        measure. The affinity of the process is set to :core:. The 
        operation to be executed and measured is given by :work:. The 
        execution time is written in shared file :wpath:, concurrent-safe 
        with :lock:.

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
    worktime = op(size) * 1000 #ms

    # Write in forks file.
    with FileLock(lock):
        wf = open(wpath, 'a')
        wf.write(f"{core} {worktime}\n")
        wf.close()

def power_fork(work, core, size):
    """
        power_fork is used to handle forked processes for energy measure.
        The affinity of the process is set to :core:. The operation to be
        executed and measured is given by :work:.
        The forked process will not return; it has to be killed by the
        parent process.
    """
    # Set core affinity.
    pid = os.getpid()
    command = "taskset -cp " + str(core) + " " + str(pid)
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)

    # Start work operation:
    op = OPERATIONS[work]
    op(size, inf=True) ## INFINITE LOOP ##

def produce_logs(workstr, time_results, power_results, logpath):
    """
        produce_logs writes the time results of :workstr: operation
        in the corresponding log files.
        A read-friendly .log file and a raw-data .csv are created.

        :workstr: str with name of operation, for log files naming.
        :results: a 2 dimension dictionary which stores execution time
                    in function of frequency and core.
    """
    # Log files paths.
    time_logpath = logpath + workstr + '.time.log'
    time_csvpath = logpath + workstr + '.time.csv'

    power_logpath = logpath + workstr + '.power.log'
    power_csvpath = logpath + workstr + '.power.csv'

    # Removing previous log files.
    if os.path.exists(time_logpath):
        os.remove(time_logpath)
    if os.path.exists(time_csvpath):
        os.remove(time_csvpath)

    if os.path.exists(power_logpath):
        os.remove(power_logpath)
    if os.path.exists(power_csvpath):
        os.remove(power_csvpath)

    # Writing log results.
    time_logf = open(time_logpath, 'w')
    time_csvf = open(time_csvpath, 'w')

    power_logf = open(power_logpath, 'w')
    power_csvf = open(power_csvpath, 'w')

    for freq in time_results:
        logfreq = freq
        if freq == NOMINAL_MAXFREQ:
            logfreq = REAL_MAXFREQ

        freqmhz = int(logfreq/1000)
        
        time_csvf.write(f"{logfreq}\n")
        power_csvf.write(f"{logfreq}\n")
        
        # TIME LOG
        time_logf.write(f"Frequency: {freqmhz} MHz\n")
        time_logf.write("-------------------------\n") # 25-
        time_logf.write("CPU   Time (ms)\n")

        for core in sorted(time_results[freq]):
            if core == -1:
                continue
            timems = time_results[freq][core]
            time_csvf.write(f"{core}, {timems}\n")
            time_logf.write(f"{core:<3}   {timems:.3f}\n")

        time_mean = time_results[freq][-1]

        time_csvf.write(f"-1, {time_mean}\n")

        time_logf.write("-------------------------\n") # 25-
        time_logf.write(f"Mean: {time_mean:.3f} ms\n")

        time_logf.write("##########################\n\n") # 25#
        
        # POWER LOG
        power_logf.write(f"Frequency: {freqmhz} MHz\n")
        power_logf.write("-------------------------\n") # 25-
        power_logf.write("Socket   Power (w)\n")

        for skt in sorted(power_results[freq]):
            if skt == -1:
                continue
            power = power_results[freq][skt]
            power_csvf.write(f"{skt}, {power}\n")
            power_logf.write(f"{skt:<6}   {power:.3f}\n")

        power_mean = power_results[freq][-1]

        power_csvf.write(f"-1, {power_mean}\n")
        
        power_logf.write("-------------------------\n")# 25-
        power_logf.write(f"Mean: {power_mean:.3f} w\n")

        power_logf.write("#########################\n\n") # 25#


#########################################################

##############
# OPERATIONS #
##############

def int_product(size, inf=False):
    """
        int_product performs a product of integer matrices and measures
        execution time. Matrices dimension is :size: x :size:.
        If :inf: is True, the matrix product is executed in an infinite
        loop.

        :size: dimension of square matrix
        :return: execution time of matrix product. 
                    If :inf: is True, never returns
    """
    # Random matrix generation.
    matA = np.random.randint(MAX_INT, size=(size, size))
    matB = np.random.randint(MAX_INT, size=(size, size))

    # ENERGY: loop.
    while(inf):
        matC = np.matmul(matA, matB)

    # TIME: operation.
    start = time.time()
    matC = np.matmul(matA, matB)
    end = time.time()

    return (end - start)

def float_product(size, inf=False):
    """
        float_product performs a product of real matrices and measures
        execution time. Matrices dimension is :size: x :size:.
        If :inf: is True, the matrix product is executed in an infinite
        loop.

        :size: dimension of square matrix
        :return: execution time of matrix product. 
                    If :inf: is True, never returns
    """
    # Random matrix generation.
    matA = np.random.rand(size, size)
    matB = np.random.rand(size, size)

    # ENERGY: loop.
    while(inf):
        matC = np.matmul(matA, matB)

    # TIME: operation.
    start = time.time()
    matC = np.matmul(matA, matB)
    end = time.time()

    return (end - start)

def int_transpose(size, inf=False):
    """
        int_transpose performs the repeated transposition of an integer matrix
        and measures execution time. Matrix dimension is :size: x :size:.
        If :inf: is True, the transposition is executed in an infinite loop.

        :size: dimension of square matrix
        :return: execution time of matrix transposition
                    If :inf: is True, never returns
    """
    # Random matrix generation.
    matA = np.random.randint(MAX_INT, size=(size, size))

    # ENERGY: loop.
    while(inf):
        matA = matA.transpose().copy()

    # TIME: operation.
    start = time.time()
    matA = matA.transpose().copy()
    matA = matA.transpose().copy()
    end = time.time()

    return (end - start)

def float_transpose(size, inf=False):
    """
        float_transpose performs the repeated transposition of a real matrix
        and measures execution time. Matrix dimension is :size: x :size:.
        If :inf: is True, the transposition is executed in an infinite loop.

        :size: dimension of square matrix
        :return: execution time of matrix transposition
                    If :inf: is True, never returns
    """
    # Random matrix generation
    matA = np.random.rand(size, size)

    # ENERGY: loop.
    while(inf):
        matA = matA.transpose().copy()

    # TIME: operation
    start = time.time()
    matA = matA.transpose().copy()
    matA = matA.transpose().copy()
    end = time.time()

    return (end - start)

def int_sort(size, inf=False):
    """
        int_sort performs the sorting of a random integer array and
        measures execution time. Array dimension is :size: * :size:.
        If :inf: is True, the sorting is executed in an infinite loop.

        :size: square-rooted dimension of array length
        :return: execution time of sorting.
                    If :inf: is True, never returns
    """
    # Random array generation
    arrayA = np.random.randint(MAX_INT, size=(size*size))

    # ENERGY: loop.
    while(inf):
        arrayB = np.sort(arrayA)

    # TIME: operation
    start = time.time()
    arrayB = np.sort(arrayA)
    end = time.time()

    return (end - start)

def float_sort(size, inf=False):
    """
        float_sort performs the sorting of a random real array and
        measures execution time. Array dimension is :size: * :size:.
        If :inf: is True, the sorting is executed in an infinite loop.

        :size: square-rooted dimension of array length
        :return: execution time of sorting.
                    If :inf: is True, never returns
    """
    # Random array generation
    arrayA = np.random.rand(size*size)

    # ENERGY: loop.
    while(inf):
        arrayB = np.sort(arrayA)

    # TIME: operation
    start = time.time()
    arrayB = np.sort(arrayA)
    end = time.time()

    return (end - start)

def int_scalar(size, inf=False):
    """
        int_scalar performs the sum of a random integer to each element of
        a random integer matrix. Matrix dimension is :size: x :size:.
        If :inf: is True, the scalar sum is executed in an infinite loop.

        :size: dimension of square matrix
        :return: execution time of scalar sum.
                    If :inf: is True, never returns
    """
    # Random matrix generation
    matA = np.random.randint(MAX_INT, size=(size, size))
    intN = np.random.randint(MAX_INT)

    # ENERGY: loop.
    while(inf):
        matB = matA + intN

    # TIME: operation.
    start = time.time()
    matB = matA + intN
    end = time.time()

    return (end - start)


def float_scalar(size, inf=False):
    """
        float_scalar performs the sum of a random real number to each
        element of a random real matrix. Matrix dimension is :size: x :size:.
        If :inf: is True, the scalar sum is executed in an infinite loop.

        :size: dimension of square matrix
        :Return: execution time of scalar sum.
                    If :inf: is True, never returns
    """
    # Randopm amtrix generation
    matA = np.random.rand(size, size)
    floatN = np.random.rand()

    # ENERGY: loop.
    while(inf):
        matB = matA + floatN

    # TIME: operation.
    start = time.time()
    matB = matA + floatN
    end = time.time()

    return (end - start)



OPERATIONS = {
        'intproduct': int_product,
        'floatproduct': float_product,
        'inttranspose': int_transpose,
        'floattranspose': float_transpose,
        'intsort': int_sort,
        'floatsort': float_sort,
        'intscalar': int_scalar,
        'floatscalar': float_scalar
        }

########################################################


def test_operation(work, freqs, sockets, mtime, size, log):
    """
        test_operation sets the environment to take measures of 
        the operation specified in :work: parameter.
        For each frequency in :freqs:, a forked process is created
        for each core in :sockets:, in which the operation is measured.
        Results are stored in a concurrent-safe file written by each
        process; then stored in the respective log files.

        :work: string that specifies the operation
        :freqs: frequencies in which the operation will be measured
        :sockets: sockets in whose cores the operation will be executed
        :mtime: wait time to measure energy consumption of operation
        :size: dimension of elements in the operation
        :log: path of log files. If None, no log files are produced
    """
    # Get implied CPU cores from sockets.
    rg = get_cores(sockets)

    # Set frequencies to minimum allowed frequency.
    # So changing the frequency implies raising it.
    minf = AVAILABLE_FREQS[0]
    lower_frequency(minf, rg)

    # Dict to store results.
    time_results = {}
    power_results = {}

    # Forks write file and lock:
    timepath = "data.temp"
    powerpath = "power.temp"
    lock = "lock.temp"
    
    # Measure exec time for each frequency in each implied core.
    freqs = sorted(freqs)
    for freq in freqs:
        raise_frequency(freq, rg)

        print(f"Measuring execution time with {int(freq/1000)} MHz.")
        
        # Remove conflicting file.
        if os.path.exists(timepath):
            os.remove(timepath)

        # Create file.
        wf = open(timepath, 'a+')
        wf.close()
        
        # Forks pid list.
        pidls = []
        for core in rg:
            pidls.append( os.fork() )
            if pidls[-1] == 0:
                time_fork(work, core, size, timepath, lock)
                exit(0)

        # Wait for all forked processes
        for pid in pidls:
            os.waitpid(pid, 0)

        # Get this frequency results.
        wf = open(timepath, 'r')
        wflines = wf.readlines()
        wf.close()
        
        time_sum = 0.0
        time_results[freq] = {}
        for line in wflines:
            corestr, worktimestr = line.split()
            core = int(corestr)
            worktime = float(worktimestr)

            time_results[freq][core] = worktime
            time_sum += worktime

        # Time mean.
        count = len(wflines)
        time_results[freq][-1] = time_sum / count

        print(f"Mean execution time: {time_results[freq][-1]:.3f} ms")

    os.remove(timepath)

    # Measure power consumption for each frequency in each socket.
    lower_frequency(minf, rg)
    for freq in freqs:
        raise_frequency(freq, rg)

        print(f"Measuring power consumption with {int(freq/1000)} MHz.")

        # Remove conflicting file.
        if os.path.exists(powerpath):
            os.remove(powerpath)

        # Create file.
        wf = open(powerpath, 'a+')
        wf.close()

        # Forks pid list.
        pidls = []
        for core in rg:
            pidls.append( os.fork() )
            if pidls[-1] == 0:
                power_fork(work, core, size)

        # pyRAPL measure of socket.
        meter = pyRAPL.Measurement(label=work)

        meter.begin()
        time.sleep(mtime)
        meter.end()

        # Kill forked processes.
        for pid in pidls:
            os.kill(pid, signal.SIGKILL)

        # Get measurements.
        timerapl = meter._results.duration

        power_sum = 0.0
        power_results[freq] = {}
        for skt in sockets:
            energyrapl = meter._results.pkg[skt]
            power = energyrapl / timerapl

            power_results[freq][skt] = power
            power_sum += power

        # Power mean.
        count = len(sockets)
        power_results[freq][-1] = power_sum / count

        print(f"Mean power consumption: {power_results[freq][-1]:.3f} W")

    os.remove(powerpath)
    os.remove(lock)

    # Writing log files
    if log is not None:
        produce_logs(work, time_results, power_results, log)
    

############################################################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)


    # Optional arguments.
    socket_help = "sockets that will be measured.\n"
    socket_help += "all sockets selected by default"
    parser.add_argument('-s', '--socket', help=socket_help, nargs='+', type=int, default=SOCKET_LIST)

    freq_help =  "FREQ frequencies in which the selected operation will be tested.\n"
    freq_help += "all available frequencies set by default"
    parser.add_argument('-f', '--freq', help=freq_help, nargs='+', type=int, default=argparse.SUPPRESS)

    dim_help = "random matrices dimension: DIM x DIM.\n"
    dim_help += "set to {} by default".format(MATRIX_SIZE)
    parser.add_argument('-d', '--dim', help=dim_help, type=int, default=MATRIX_SIZE)

    time_help = "time (seconds) spent in measuring energy consumption for each frequency"
    time_help += "set to {} seconds by default".format(MEASURE_TIME)
    parser.add_argument('-t', '--time', help=time_help, type=int, default=MEASURE_TIME)

    log_help = "produces log files in LOG path where results are stored.\n"
    log_help += "set to '{}' by default.".format(LOG_PATH)
    parser.add_argument('-l', '--log', help=log_help, type=str, nargs='?', const=LOG_PATH, default=argparse.SUPPRESS)
    
    # Positional arguments.
    work_help = "numpy action to be tested: intproduct transpose sort"
    parser.add_argument('work', help=work_help)

    ##intprod_help = "multiplication of random integer matrices" 
    ##trans_help = "transposal of random matrix"
    ##sort_help = "sorting of random list"

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

    # Gets closest frequencies to the selected ones.
    freqs = []
    if 'freq' in args:
        userfs = args.freq
        for freq in userfs:
            freqs.append( closest_frequency(freq * 1000) )
    else:
        freqs = AVAILABLE_FREQS

    size = args.dim
    mtime = args.time

    # Set log flag.
    log = None
    if 'log' in args:
        log = args.log

    # TODO clearly simplify.
    if args.work == 'intproduct':
        test_operation('intproduct', freqs, sockets, mtime, size, log)
    if args.work == 'floatproduct':
        test_operation('floatproduct', freqs, sockets, mtime, size, log)

    if args.work == 'inttranspose':
        test_operation('inttranspose', freqs, sockets, mtime, size, log)
    if args.work == 'floattranspose':
        test_operation('floattranspose', freqs, sockets, mtime, size, log)

    if args.work == 'intsort':
        test_operation('intsort', freqs, sockets, mtime, size, log)
    if args.work == 'floatsort':
        test_operation('floatsort', freqs, sockets, mtime, size, log)
                
    if args.work == 'intscalar':
        test_operation('intscalar', freqs, sockets, mtime, size, log)
    if args.work == 'floatscalar':
        test_operation('floatscalar', freqs, sockets, mtime, size, log)

if __name__ == '__main__':
    main()
