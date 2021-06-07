import argparse
import pyRAPL
import os
import cpufreq
import signal
import time
from filelock import FileLock

import powerutil.numpypower as npp
import timeutil.numpytime as npt

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

def rapl_power(label, powertime, sockets):
    meter = pyRAPL.Measurement(label=label)

    while meter._results is None or meter._results.pkg is None:
        meter.begin()
        time.sleep(powertime)
        meter.end()

    results = {}
    m_time = meter._results.duration # micro-s
    for skt in sockets:
        m_energy = meter._results.pkg[skt] # micro-J
        results[skt] = m_energy / m_time # watts

    return results


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
            > time = time_results[core][freq]
        logpath : str
            Path of folder where log files will be generated.
    """
    # Log files paths.
    time_logpath = logpath + workstr + '.time.log'
    time_csvpath = logpath + workstr + '.time.csv'

    time_csv(time_results, time_csvpath)
    time_log(time_results, time_logpath)

def time_csv(results, csvpath):
    cores = sorted(results.keys())
    freqs = results[-1].keys()
    
    # Create or overwrite.
    if os.path.exists(csvpath):
        os.remove(csvpath)
    csvf = open(csvpath, 'w')

    # Header.
    csvf.write("Frequency")
    for core in cores:
        csvf.write(f",{core},")
    csvf.write("\n")

    # Results.
    for freq in freqs:
        csvf.write(f"{freq}")
        for core in cores:
            csvf.write(f",{results[core][freq]},")
        csvf.write("\n")

    csvf.close()

def time_log(results, logpath):
    cores = sorted(results.keys())
    freqs = results[-1].keys()

    # Create or overwrite.
    if os.path.exists(logpath):
        os.remove(logpath)
    logf = open(logpath, 'w')

    # Results
    for freq in freqs:
        if freq == NOMINAL_MAXFREQ:
            logfreq = "MAX"
        else:
            logfreq = freq // 1000 # MHz

        logf.write(f"Frequency (MHz): {logfreq}\n")
        logf.write("-------------------------\n") # 25-
        logf.write("CPU   Time (ms)\n")

        for core in cores:
            if core == -1:
                continue
            timems = time_results[core][freq]
            logf.write(f"{core:<3}   {timems:.3f}\n")

        mean = time_results[-1][freq]

        logf.write("-------------------------\n") # 25-
        logf.write(f"Mean: {mean:.3f} ms\n")

        logf.write("##########################\n\n") # 25#

    logf.close()
    

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
            > power = power_results[skt][freq]
        logpath : str
            Path of folder where log files will be generated.
    """
    # Log files paths.
    power_logpath = logpath + workstr + '.power.log'
    power_csvpath = logpath + workstr + '.power.csv'

    power_csv(power_results, power_csvpath)
    power_log(power_results, power_logpath)

def power_csv(results, csvpath):
    sockets = sorted(results.keys())
    freqs = results[-1].keys()
    
    # Create or overwrite.
    if os.path.exists(csvpath):
        os.remove(csvpath)
    csvf = open(csvpath, 'w')

    # Header.
    csvf.write("Frequency")
    for skt in sockets:
        csvf.write(f",{skt}")
    csvf.write("\n")

    # Results.
    for freq in freqs:
        csvf.write(f"{freq}")
        for skt in sockets:
            csvf.write(f",{results[skt][freq]}")
        csvf.write("\n")

    csvf.close()

def power_log(results, logpath):
    sockets = sorted(results.keys())
    freqs = results[-1].keys()

    # Create or overwrite.
    if os.path.exists(logpath):
        os.remove(logpath)
    logf = open(logpath, 'w')

    # Results
    for freq in freqs:
        if freq == NOMINAL_MAXFREQ:
            logfreq = "MAX"
        else:
            logfreq = freq // 1000 # MHz

        logf.write(f"Frequency (MHz): {logfreq}\n")
        logf.write("-------------------------\n") # 25-
        logf.write("Socket   Power (w)\n")

        for skt in sockets:
            if skt == -1:
                continue
            power = results[skt][freq]
            logf.write(f"{skt:<6}   {power:.3f}\n")

        mean = results[-1][freq]

        logf.write("-------------------------\n") # 25-
        logf.write(f"Mean: {mean:.3f} ms\n")

        logf.write("##########################\n\n") # 25#

    logf.close()


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
    time_results[-1] = {}
    for core in cores:
        time_results[core] = {}

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
                time_fork(work, core, size, rep, timepath, lockpath)
                exit(0)

        # Wait for all forked processes
        for pid in pidls:
            os.waitpid(pid, 0)

        # Get this frequency results.
        wf = open(timepath, 'r')
        wflines = wf.readlines()
        wf.close()
        
        time_sum = 0.0
        for line in wflines:
            corestr, worktimestr = line.split()
            core = int(corestr)
            worktime = float(worktimestr)

            time_results[core][freq] = worktime
            time_sum += worktime

        # Time mean.
        count = len(wflines)
        time_results[-1][freq] = time_sum / count

        print(f"Mean execution time: {time_results[-1][freq]:.3f} ms")

    os.remove(timepath)
    os.remove(lockpath)

    # Writing log files
    if log is not None:
        time_logs(work, time_results, log)

def power_measure(work, freqs, sockets, size, powertime, log):
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
        sockets : list
            List of sockets where the operation will be executed.
        size : int
            Dimension of the Numpy elements used in the operation.
        powertime : float
            Time the pyRAPL utility will spend measuring energy consumption.
        log : str
            Path of the folder where log files will be generated.
            If None, log files will not be produced.
    """
    cores = get_cores(sockets)

    # Set frequencies to minimum allowed frequency.
    # so that changing the frequency implies raising it.
    minf = AVAILABLE_FREQS[0]
    lower_frequency(minf, cores)

    # Dict to store results
    power_results = {}
    power_results[-1] = {}
    for skt in sockets:
        power_results[skt] = {}

    # Measure execution time for each frequency in each implied core.
    freqs = sorted(freqs)
    for freq in freqs:
        raise_frequency(freq, cores)

        print(f"Measuring energy consumption with {freq // 1000} MHz.")

        # Forks pid list
        pidls = []
        for core in cores:
            pidls.append( os.fork() )
            if pidls[-1] == 0:
                power_fork(work, core, size)
                exit(0)

        # Measure.
        freqpower = rapl_power(work, powertime, sockets)
        for skt in freqpower:
            power_results[skt][freq] = freqpower[skt]

        # Wait for all forked processes
        for pid in pidls:
            os.kill(pid, signal.SIGKILL)

        # Time mean.
        count = len(freqpower)
        power_sum = sum(freqpower.values())
        power_results[-1][freq] = power_sum / count

        print(f"Mean execution power: {power_results[-1][freq]:.3f} w")

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
        '--time', help=time_help,
        action='store_true'
    )

    power_help = "Measures energy consumption of the specified operation."
    metrics.add_argument(
        '--power', help=power_help,
        action='store_true'
    )

    ## Execution: cores or sockets
    cpucores = parser.add_mutually_exclusive_group(required=True)

    cores_help = "The operation is executed in each specified core."
    cpucores.add_argument(
        '-c', '--cores', metavar='cores', help=cores_help,
        nargs='+',
        type=int,
        default=argparse.SUPPRESS
    )

    sockets_help = "The operation is executed in the cores of the specified "
    sockets_help += "sockets."
    cpucores.add_argument(
        '-s', '--sockets', metavar='sockets', help=sockets_help,
        nargs='+',
        type=int,
        default=argparse.SUPPRESS
    )

    ## Measure process affinity: cores or sockets
    affinity = parser.add_mutually_exclusive_group()

    affcores_help = "The cores in which the measure process will be processed."
    affinity.add_argument(
        '--affcores', metavar='affcores', help=affcores_help,
        nargs='+',
        type=int,
        default=argparse.SUPPRESS
    )

    affsockets_help = "The sockets in whose cores the measure process will be "
    affsockets_help += "processed."
    affinity.add_argument(
        '--affsockets', metavar='affsockets', help=affsockets_help,
        nargs='+',
        type=int,
        default=argparse.SUPPRESS
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
        '-r', '--rep', metavar='rep', help=rep_help,
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
        default=None
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

    # Measure process affinity
    if 'affcores' in args:
        os.sched_setaffinity(0, args.affcores)
    elif 'affsockets' in args:
        os.sched_setaffinity(0, get_cores(args.affsockets))

    # Gets closest frequencies to the selected ones.
    freqs = []
    if 'freqs' in args:
        userfs = args.freqs
        for freq in userfs:
            freqs.append( closest_frequency(freq * 1000) )
    else:
        freqs = AVAILABLE_FREQS

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

        power_measure(args.work, freqs, args.sockets, args.dim, args.powertime, args.log)

    if args.time:
        # Get cores
        cores = []
        if 'cores' in args:
            cores = args.cores
        elif 'sockets' in args:
            cores = get_cores(args.sockets)

        time_measure(args.work, freqs, cores, args.dim, args.rep, args.log)


if __name__ == '__main__':
    main()
