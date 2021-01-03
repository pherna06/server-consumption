import argparse
import cpufreq

#########################
### CPUFREQ FUNCTIONS ###
#########################

def get_governors(rg = None):
    """
        Displays the current governor scheme for each desired online CPU core.

        Parameters
        ----------
        rg : int, list
            An integer or list of integers with the indices of CPU cores whose
            governors are to be displayed. If None, all online cores will be
            displayed.
    """
    # Parameter parsing.
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _cpu.get_online_cpus()

    # Fetch all CPU governors.
    govs = _cpu.get_governors()
    
    # Display desired cores governor schemes.
    print("CPU   Governor")
    for core in rg:
        # It won't display unrecognised cores.
        if core not in govs:
            continue
        
        gov = govs[core]
        print(f"{core:<3}   {gov:<12}")


def set_governors(gov, rg = None):
    """
        Sets the desired online CPU cores to a specific governor scheme.

        Parameters
        ----------
        gov : str
            The governor scheme to which the cores will be set to.
        rg : int, list
            An integer or list of integers with the indices of CPU cores whose
            governor scheme will be modified. If None, all online cores will be
            modified.
    """
    if gov not in _available_govs:
        # Handle 'not available governor' verbose.
        print("ERROR: Not a valid governor. Please, try one of the following:")
        for av_gov in _available_govs:
            print(av_gov)
    else:
        # Parameter parsing.
        if isinstance(rg, int):
            rg = [rg]
        elif rg is None:
            rg = _cpu.get_online_cpus()

        # Governor modification.
        for core in rg:
            try:
                _cpu.set_governors(gov, rg)
                if _verbose:
                    print(f"CPU {core} set to {gov}.")
            except:
                print(f"ERROR: An exception occurred. Check if CPU {core} exists.")


def get_online():
    """
        Displays all online CPU cores.
    """
    print( "Online CPUs:" + "".join( f" {cpu}" for cpu in _cpu.get_online_cpus() ) )


def set_online(rg = None):
    """
        Sets the desired CPU cores online.

        Parameters
        ----------
        rg : int, list
            An integer or list of integers with the indices of CPU cores that
            will be set online. If None, all CPU cores will be set online.
    """
    if isinstance(rg, int):
        rg = [rg]

    if rg is None:
        _cpu.enable_all_cpu()
        print("All CPUs enabled.")
    else:
        for core in rg:
            try:
                _cpu.enable_cpu(core)
                if _verbose:
                    print(f"CPU {core} set online.")
            except:
                print(f"ERROR: An exception occurred. Check if CPU {core} exists.")


def set_offline(rg = None):
    """
        Sets the desired CPU cores offline.

        WARNING: Be careful to set all CPU cores offline, though OS will
        probably force at least one (typically 0) to stay awake.

        Parameters
        ----------
        rg : int, list
            An integer or list of integers with the indices of CPU cores that
            will be set offline. If None, all online CPU cores will be set 
            offline.
    """
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _cpu.get_online_cpus()

    for core in rg:
        try:
            _cpu.disable_cpu(core)
            if _verbose:
                print(f"CPU {core} set offline.")
        except:
            print(f"ERROR: An exception occurred. Check if CPU {core} exists.")


def set_minimum(freq, rg = None):
    """
        Sets the desired online CPU cores minimum frequency.

        Parameters
        ----------
        freq : int
            The desired minimum frequency, in KHz. It must be over the CPU
            overall minimum frequency; an exception will be raised otherwise.
        rg : int, list
            An integer or list of integers with the indices of CPU cores whose
            minimum frequency will be modified. If None, all online CPU cores
            will be modified.
    """
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _cpu.get_online_cpus()

    for core in rg:
        try:
            _cpu.set_min_frequencies(freq, core)
            if _verbose:
                print(f"CPU {core} minimum frequency set to {int(freq/1000)} MHz.")
        except Exception as e:
            print(f"ERROR: An exception occurred for CPU {core}.")
            print(e)


def set_maximum(freq, rg = None):
    """
        Sets the desired online CPU cores maximum frequency.

        Parameters
        ----------
        freq : int
            The desired maximum frequency, in KHz. It must be below the CPU
            overall maximum frequency; an exception will be raised otherwise.
        rg : int, list
            An integer or list of integers with the indices of CPU cores whose
            maximum frequency will be modified. If None, all online CPU cores
            will be modified.
    """
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _cpu.get_online_cpus()

    for core in rg:
        try:
            _cpu.set_max_frequencies(freq, core)
            if _verbose:
                print(f"CPU {core} maximum frequency set to {int(freq/1000)} MHz.")
        except Exception as e:
            print(f"ERROR: An exception occurred for CPU {core}.")
            print(e)


def set_frequency(freq, rg = None, limits = True):
    """
        Sets the desired online CPU cores to a specific frequency.

        Parameters
        ----------
        freq : int
            The specified frequency, in KHz and within _available_freqs.
        rg : int, list
            An integer or list of integers with the indices of CPU cores whose
            frequency will be modified. If None, all online CPU cores will be
            modified.
        limits : bool
            If True, the minimum and maximum frequencies of the affected cores
            will also be set to the specified frequency. This is made to ensure
            the that core frequency is 'physically' set to that value, which
            will not occur in some CPUs if this is not done.
            If False, frequency will only be modified with cpufreq's 
            set_frequencies() function, which does not modify minimum nor 
            maximum frequencies. It will raise an error if the specified 
            frequency is out of those limits.

    """
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _cpu.get_online_cpus()

    for core in rg:
        try:
            # We ensure that frequency is properly modified
            if limits:
                minf = _cpu.get_min_freq()[core]
                if minf < freq:
                    _cpu.set_max_frequencies(freq, core)
                    _cpu.set_min_frequencies(freq, core)
                else:
                    _cpu.set_min_frequencies(freq, core)
                    _cpu.set_max_frequencies(freq, core)

            _cpu.set_frequencies(freq, core)
            
            if _verbose:
                print(f"CPU {core} frequency set to {int(freq/1000)} MHz.")
        except:
            print(f"ERROR: An exception occurred. Check if CPU {core} exists.")


def get_frequency(rg = None):
    """
        Displays minimum, maximum and current 'physical' frequency of the 
        desired online CPU cores.

        Parameters
        ----------
        rg : int, list
            An integer or list of integers with the indices of CPU cores whose
            frequency data is to be displayed. If None, all online CPU cores
            will be displayed.
    """
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _cpu.get_online_cpus()

    freqs = _cpu.get_frequencies()
    minfs = _cpu.get_min_freq()
    maxfs = _cpu.get_max_freq()

    print("CPU   Current (MHz)   Minimum (MHz)   Maximum (MHz)")
    for core in rg:
        freq = int(freqs[core]/1000) if core in freqs else '???'
        minf = int(minfs[core]/1000) if core in minfs else '???'
        maxf = int(maxfs[core]/1000) if core in maxfs else '???'
        print(f"{core:<3}   {freq:<13}   {minf:<13}   {maxf:<13}")


#################
### UTILITIES ###
#################

def closest_frequency(freq):
    """
        Approximates the specified frequency to the closest higher or equal 
        available frequency.

        Parameters
        ----------
        freq : int
            The specified frequency, in KHz, to be approximated. Must not be
            smaller than the minimum available_frequency.

        Returns
        -------
        int
            The minimum frequency of _available_freqs which is higher or equal
            than the given frequency.

    """
    if freq < _available_freqs[0]:
        print("ERROR: Specified frequency is below the minimum allowed frequency.")
        exit()

    av_freq = _available_freqs[0]
    for af in _available_freqs:
        if freq <= av_freq:
            break
        av_freq = af

    return av_freq


def show_list(mode):
    """
        Prints the available values of the specified mode: freq or gov.

        Parameters
        ----------
        mode : str
            The specified mode whose available values will be displayed.
    """
    if mode == 'freq':
        for freq in _available_freqs:
            print(freq // 1000)

    if mode == 'gov':
        for gov in _available_govs:
            print(gov)

#########################
### COMMAND INTERFACE ###
#########################

def get_parser():
    # Parser description and creation.
    desc = "A command interface that implements the functionalities of Python "
    desc += "cpufreq module."
    parser = argparse.ArgumentParser(description = desc)

    # Parser argument to select CPU cores.
    cpu_help = "CPU cores to watch or modify. "
    cpu_help += "All (online) cores selected by default."
    parser.add_argument(
        '-c', '--cpu', metavar='CPU', help=cpu_help, 
        nargs='+', 
        type=int, 
        default=None
    )

    # Parser argument to activate verbose.
    verbose_help = "Displays verbose for actions that modify the CPU state."
    parser.add_argument(
        '-v', '--verbose', help=verbose_help,
        action='store_true'
    )

    actions = parser.add_mutually_exclusive_group()

    # Argument for handling governor schemes.    
    gov_help = "'show' displays actual CPU governor schemes. "
    gov_help += "To change CPU governor scheme, use one of the following: "
    gov_help += "".join(f" {gov}" for gov in _available_govs)
    actions.add_argument(
        '-g', '--governor', metavar='GOV', help=gov_help, 
        type=str,
        default=argparse.SUPPRESS
    )
    

    # Argument for handling online setting.
    online_help = "'show' displays online CPUs. "
    online_help += "'on' enables selected CPUs. "
    online_help += "'off' disables selected CPUs. "
    actions.add_argument(
        '-o', '--online', metavar='MODE', help=online_help,
        type=str,
        default=argparse.SUPPRESS
    )

    # Argument for setting minimum frequency. 
    min_help = "sets CPUs minimum frequency to MIN (in MHz)."
    actions.add_argument(
        '--min', metavar='MIN', help=min_help, 
        type=int,
        default=argparse.SUPPRESS
    )

    # Argument for setting maximum frequency.
    max_help = "sets CPUs maximum frequency to MAX (in MHz)."
    actions.add_argument(
        '--max', metavar='MAX', help=max_help, 
        type=int,
        default=argparse.SUPPRESS
    )

    # Argument for setting CPU frequency.
    freq_help = "sets CPUs frequency to FREQUENCY (in MHz)."
    actions.add_argument(
        '-f', '--frequency', metavar='FREQ', help=freq_help, 
        type=int,
        default=argparse.SUPPRESS
    )

    # Argument for setting CPU frequency without modifying min/max frequencies.
    nolimits_help = "sets CPUs frequency to NOLIMITS (in MHz). "
    nolimits_help += "Minimum and maximum frequencies are not modified."
    actions.add_argument(
        '--nolimits', metavar='FREQ', help=nolimits_help, 
        type=int,
        default=argparse.SUPPRESS
    )
        
    # Argument for showing CPU frequencies.
    show_help = "show CPUs frequency information."
    actions.add_argument(
        '-s', '--show', help=show_help, 
        action='store_true'
    )

    # Argument for CPU reset to default state.
    reset_help = "resets CPU to its default state."
    actions.add_argument(
        '-r', '--reset', help=reset_help,
        action='store_true'
    )

    # Argument for showing available values.
    list_help = "'freq' shows available frequencies. "
    list_help += "'gov' shows available governors."
    actions.add_argument(
        '-l', '--list', metavar='MODE', help=list_help,
        type=str,
        default=argparse.SUPPRESS
    )

    return parser


def main():
    # 'cpufreq' module object to modify CPU states.
    global _cpu
    _cpu = cpufreq.cpuFreq()

    # List of available CPU governor schemes.
    global _available_govs
    _available_govs = [
            'conservative',
            'ondemand',
            'userspace',
            'powersave',
            'performance',
            'schedutil'
            ]

    # List of CPU available frequencies.
    global _available_freqs
    _available_freqs = sorted(_cpu.available_frequencies)

    # Verbose option.
    global _verbose

    ## Command parsing.
    parser = get_parser()
    args = parser.parse_args()

    rg = args.cpu
    _verbose = True if args.verbose else False
    
    if 'governor' in args:
        gov = args.governor
        
        if gov == 'show':
            get_governors(rg)
        else:
            set_governors(gov, rg)
    
    if 'online' in args:
        online = args.online

        if online == 'show':
            get_online()
        if online == 'on':
            set_online(rg)
        if online == 'off':
            set_offline(rg)

    if 'min' in args:
        minf = closest_frequency(args.min * 1000)
        set_minimum(minf, rg)

    if 'max' in args:
        maxf = closest_frequency(args.max * 1000)
        set_maximum(maxf, rg)

    if 'frequency' in args:
        freq = closest_frequency(args.frequency * 1000)
        set_frequency(freq, rg)

    if 'nolimits' in args:
        freq = closest_frequency(args.nolimits * 1000)
        set_frequency(freq, rg, False)

    if 'list' in args:
        show_list(args.list)

    if args.show:
        get_frequency(rg)

    if args.reset:
        _cpu.reset()


if __name__ == '__main__':
    main()
