import argparse
from numpy.testing._private.utils import requires_memory
import pyRAPL
import time
import subprocess
import os
import signal
import cpufreq
from filelock import FileLock

import numpypower as npp
import numpytime as npt

# MODIFY ACCORDING TO YOUR MACHINE CPU CONFIGURATION.
CPU_LIST = list(range(16))
SOCKET_LIST = [0, 1]
SOCKET_DICT = {
        0: CPU_LIST[0:8], 
        1: CPU_LIST[8:16]
}

# cpufreq variables.
AVAILABLE_FREQS = sorted(cpufreq.cpuFreq().available_frequencies)

## TIME TEST
TIME_CPU = 0
TIME_REPS = 10

# ENERGY MEASURE
DEF_POWERTIME = 2.0

# MAX FREQUENCY
NOMINAL_MAXFREQ = 2601000
REAL_MAXFREQ = 3000000

# DIMENSION
DEF_DIM = 1000

# REPEAT
DEF_REP = 1

#####################
# UTILITY FUNCTIONS #
#####################

def closest_frequency(freq):
    """
        Calculates the minimum available frequency which is greater or equal 
        than the given frequency. If given frequency is below the minimum 
        allowed frequency an error is raised.

        Parameters
        ----------
        freq : int
            The given frequency in KHz.
        
        Returns
        -------
        int
            The minimum frequency greater or equal than the provided one.
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
        Sets the specified cores to the given frequency, supposed that their
        actual frequencies are greater than the given one.

        Parameters
        ----------
        freq : int
            The frequency, in KHz, to which cores will be set.
        rg : list
            The cores whose frequency will be lowered.
    """
    cpu = cpufreq.cpuFreq()
    cpu.set_min_frequencies(freq, rg)
    cpu.set_max_frequencies(freq, rg)
    cpu.set_frequencies(freq, rg)

def raise_frequency(freq, rg):
    """
        Sets the specified cores to the given frequency, supposed that their
        actual frequencies are lower than the given one.

        Parameters
        ----------
        freq : int
            The frequency, in KHz, to which cores will be set.
        rg : list
            The cores whose frequency will be raised.
    """
    cpu = cpufreq.cpuFreq()
    cpu.set_max_frequencies(freq, rg)
    cpu.set_min_frequencies(freq, rg)
    cpu.set_frequencies(freq, rg)

def get_cores(sockets):
    """
        Retrieves the cores associated to the given socket IDs.

        Parameters
        ----------
        sockets : list
            The specified socket numbers.
        
        Returns
        -------
        list
            The cores associated with the given socket numbers.
    """
    rg = []
    for skt in sockets:
        rg.extend(SOCKET_DICT[skt])

    return rg


######################
### FORK FUNCTIONS ###
######################

def time_fork(work, core, size, rep, wpath, lock):
    """
        Handles forked processes for execution time measurement. The affinity 
        of the process is set to the given core. Execution times are stored
        in a concurrent-safe shared file.

        Parameters
        ----------
        work : str
            Name that specifies the operation to be measured.
        core : int
            CPU core to set process affinity.
        size : int
            Dimension of the Numpy elements used in the operation.
        rep : int
            Number of iterations of the operation to be measured.
        wpath : str
            Path of write file shared by the forked processes.
        lock : str
            Path of lock for write file safe-concurrency.
    """
    # Set core affinity.
    os.sched_setaffinity(0, {core})

    # Pick work operation:
    worktime = npt.timeop(op=work, config={'size': size, 'rep': rep}) * 1000 # ms

    # Write in forks file.
    with FileLock(lock):
        wf = open(wpath, 'a')
        wf.write(f"{core} {worktime}\n")
        wf.close()

def power_fork(work, core, size):
    """
        Handles forked processes for operation energy measurement. The affinity
        of the process is set to the given core. The forked process will not 
        return; it has to be killed by the parent process.

        Parameters
        ----------
        work : str
            Name that specifies the operation to be measured.
        core : int
            CPU core to set process affinity.
        size : int
            Dimension of the Numpy elements used in the operation.
    """
    # Set core affinity.
    os.sched_setaffinity(0, {core})

    # Start work operation:
    npp.powerop(op=work, config={'size': size})


######################
### LOG GENERATION ###
######################

def time_logs(workstr, time_results, logpath):
    """
        Saves the time results of an operation in the specified log files. A 
        read-friendly .log file and a raw-data .csv are created.

        Parameters
        ----------
        workstr : str
            Name of the operation, used for log files naming.
        time_results : dict
            Dictionary with execution times for frequency and core.
            > time = time_results[freq][core]
        logpath : str
            Path of folder where log files will be generated.
    """
    # Log files paths.
    time_logpath = logpath + workstr + '.time.log'
    time_csvpath = logpath + workstr + '.time.csv'

    # Removal of previous log files.
    if os.path.exists(time_logpath):
        os.remove(time_logpath)
    if os.path.exists(time_csvpath):
        os.remove(time_csvpath)

    # Writing log results.
    time_logf = open(time_logpath, 'w')
    time_csvf = open(time_csvpath, 'w')

    for freq in time_results:
        logfreq = freq
        if freq == NOMINAL_MAXFREQ:
            logfreq = REAL_MAXFREQ

        freqmhz = int(logfreq/1000)
        
        time_csvf.write(f"{logfreq}\n")
        
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


def power_logs(workstr, power_results, logpath):
    """
        Saves the power results of an operation in the specified log files. A 
        read-friendly .log file and a raw-data .csv are created.

        Parameters
        ----------
        workstr : str
            Name of the operation, used for log files naming.
        power_results : dict
            Dictionary with energy consumption for frequency and core.
            > power = power_results[freq][core]
        logpath : str
            Path of folder where log files will be generated.
    """
    # Log files paths.
    power_logpath = logpath + workstr + '.power.log'
    power_csvpath = logpath + workstr + '.power.csv'

    # Removal of previous log files.
    if os.path.exists(power_logpath):
        os.remove(power_logpath)
    if os.path.exists(power_csvpath):
        os.remove(power_csvpath)

    # Writing log results.
    power_logf = open(power_logpath, 'w')
    power_csvf = open(power_csvpath, 'w')

    for freq in power_results:
        logfreq = freq
        if freq == NOMINAL_MAXFREQ:
            logfreq = REAL_MAXFREQ

        freqmhz = int(logfreq/1000)
        
        power_csvf.write(f"{logfreq}\n")
        
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


###################
### MEASUREMENT ###
###################

def time_measure(work, freqs, cores, size, rep, log):
    """
        Sets the environment to measure the execution times of the specified
        operation. For each frequency and core, a forked process is created in
        which the operation time is measured. Execution times are written on
        a concurrent-safe shared temporal file. Log files are generated
        if indicated.

        Parameters
        ----------
        work : str
            Name that specifies the operation to be measured.
        freqs : list
            List of frequencies (in KHz) in which the execution time will be
            measured.
        cores : list
            List of cores where the operation will be executed.
        size : int
            Dimension of the Numpy elements used in the operation.
        rep : int
            Number of iterations of the operation to be measured.
        log : str
            Path of the folder where log files will be generated.
            If None, log files will not be produced.
    """
    # Set frequencies to minimum allowed frequency.
    # so that changing the frequency implies raising it.
    minf = AVAILABLE_FREQS[0]
    lower_frequency(minf, cores)

    # Dict to store results
    time_results = {}

    # Shared temporal file and lock:
    timepath = 'time.temp'
    lockpath = 'lock.temp'

    # Measure execution time for each frequency in each implied core.
    freqs = sorted(freqs)
    for freq in freqs:
        raise_frequency(freq, cores)

        print(f"Measuring execution time with {freq // 1000} MHz.")

        # Remove conflicting file.
        if os.path.exists(timepath):
            os.remove(timepath)

        # Create file.
        wf = open(timepath, 'a+')
        wf.close()

        # Forks pid list
        pidls = []
        for core in cores:
            pidls.append( os.fork() )
            if pidls[-1] == 0:
                time_fork(work, core, size, timepath, lockpath)
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

    # Writing log files
    if log is not None:
        time_logs(work, time_results, log)

def power_measure(work, freqs, cores, size, log):
    """
        Sets the environment to measure the energy consumption of the specified
        operation. For each frequency and core, a forked process is created in
        which the operation power is measured. Measured powers are written on
        a concurrent-safe shared temporal file. Log files are generated
        if indicated.

        Parameters
        ----------
        work : str
            Name that specifies the operation to be measured.
        freqs : list
            List of frequencies (in KHz) in which the execution time will be
            measured.
        cores : list
            List of cores where the operation will be executed.
        size : int
            Dimension of the Numpy elements used in the operation.
        log : str
            Path of the folder where log files will be generated.
            If None, log files will not be produced.
    """
    # Set frequencies to minimum allowed frequency.
    # so that changing the frequency implies raising it.
    minf = AVAILABLE_FREQS[0]
    lower_frequency(minf, cores)

    # Dict to store results
    power_results = {}

    # Shared temporal file and lock:
    powerpath = 'power.temp'
    lockpath = 'lock.temp'

    # Measure execution time for each frequency in each implied core.
    freqs = sorted(freqs)
    for freq in freqs:
        raise_frequency(freq, cores)

        print(f"Measuring energy consumption with {freq // 1000} MHz.")

        # Remove conflicting file.
        if os.path.exists(powerpath):
            os.remove(powerpath)

        # Create file.
        wf = open(powerpath, 'a+')
        wf.close()

        # Forks pid list
        pidls = []
        for core in cores:
            pidls.append( os.fork() )
            if pidls[-1] == 0:
                power_fork(work, core, size, powerpath, lockpath)
                exit(0)

        # Wait for all forked processes
        for pid in pidls:
            os.waitpid(pid, 0)

        # Get this frequency results.
        wf = open(powerpath, 'r')
        wflines = wf.readlines()
        wf.close()
        
        time_sum = 0.0
        power_results[freq] = {}
        for line in wflines:
            corestr, worktimestr = line.split()
            core = int(corestr)
            worktime = float(worktimestr)

            power_results[freq][core] = worktime
            time_sum += worktime

        # Time mean.
        count = len(wflines)
        power_results[freq][-1] = time_sum / count

        print(f"Mean execution time: {power_results[freq][-1]:.3f} ms")

    os.remove(powerpath)

    # Writing log files
    if log is not None:
        power_logs(work, power_results, log)


#########################
### COMMAND INTERFACE ###
#########################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    ## Metrics: time or power.
    metrics = parser.add_mutually_exclusive_group(required=True)

    time_help = "Measures execution time of the specified operation."
    metrics.add_argument(
        '--time', metavar='time', help=time_help,
        action='store_true'
    )

    power_help = "Measures energy consumption of the specified operation."
    metrics.add_argument(
        '--power', metavar='power', help=power_help,
        action='store_true'
    )

    ## Execution: cores or sockets
    cpucores = parser.add_mutually_exclusive_group(required=True)

    cores_help = "The operation is executed in each specified core."
    cpucores.add_argument(
        '-c', '--cores', metavar='cores', help=cores_help,
        nargs='+',
        type=int,
    )

    sockets_help = "The operation is executed in the cores of the specified "
    sockets_help += "sockets."
    cpucores.add_argument(
        '-s', '--sockets', metavar='sockets', help=sockets_help,
        nargs='+',
        type=int
    )

    # Optional arguments
    ## Frequencies
    freq_help =  "FREQS frequencies in which the selected operation will be "
    freq_help += "tested. All available frequencies set by default"
    parser.add_argument(
        '-f', '--freqs', metavar='freqs', help=freq_help, 
        nargs='+', 
        type=int, 
        default=argparse.SUPPRESS
    )

    ## Dimension
    dim_help = "Dimension used for the Numpy elements in the operation."
    dim_help += "For matrices, size is DIM X DIM."
    dim_help += "Default value for DIM is {}".format(DEF_DIM)
    parser.add_argument(
        '-d', '--dim', metavar='dim', help=dim_help, 
        type=int, 
        default=DEF_DIM
    )

    ## Repetitions
    rep_help = "Number of iterations of the selected operation to obtain mean "
    rep_help += "execution time."
    rep_help += "Default value is {}".format(DEF_REP)
    parser.add_argument(
        '-r', '--rep', metavar='rep', help='rep_help',
        type=int,
        default=DEF_REP
    )

    ## Energy measure time
    powertime_help = "Time (in seconds) to be spent in measuring energy "
    powertime_help += "consumption for each frequency"
    powertime_help += "Default value is {}".format(DEF_POWERTIME)
    parser.add_argument(
        '-t', '--powertime', metavar='powertime', help=powertime_help,
        type=float,
        default=DEF_POWERTIME
    )

    ## Log files path
    log_help = "Path to folder where log files will be generated."
    log_help += "If not specified, log files will not be produced"
    parser.add_argument(
        '-l', '--log', metavar='log', help=log_help,
        nargs='?',
        type=str,
        default=argparse.SUPPRESS
    )

    # Positional arguments.
    work_help = "Name of operation to be tested: intproduct inttranspose "
    work_help += "intsort intscalar floatproduct floattranspose floatsort "
    work_help += "floatscalar"
    parser.add_argument('work', help=work_help)

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Socket must be selected for pyRAPL measurement.
    if args.power:
        if 'sockets' not in args:
            print("ERROR: sockets must be selected to measure power.")
            exit()

        try:
            pyRAPL.setup(
                devices = [pyRAPL.Device.PKG],
                socket_ids = args.sockets
            )
        except:
            print("ERROR: check if selected sockets exist.")
            exit()

    # Get cores
    cores = []
    if 'cores' in args:
        cores = args.cores
    elif 'sockets' in args:
        cores = get_cores(args.sockets)

    # Gets closest frequencies to the selected ones.
    freqs = []
    if 'freq' in args:
        userfs = args.freq
        for freq in userfs:
            freqs.append( closest_frequency(freq * 1000) )
    else:
        freqs = AVAILABLE_FREQS

    # Set log flag.
    log = None
    if 'log' in args:
        log = args.log

    # Measurement
    if args.time:
        time_measure(args.work, freqs, cores, args.dim, args.rep, log)

    if args.power:
        power_measure(args.work, freqs, cores, args.dim, log)


if __name__ == '__main__':
    main()
