import argparse
import cpufreq

_available_govs = [
        'conservative',
        'ondemand',
        'userspace',
        'powersave',
        'performance',
        'schedutil'
        ]

_available_cpus = cpufreq.cpuFreq().get_online_cpus()

_available_freqs = sorted(cpufreq.cpuFreq().available_frequencies)


def get_governors(rg = None):
    govs = cpufreq.cpuFreq().get_governors()
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _available_cpus
    
    print("CPU   Governor")
    for core in rg:
        gov = govs[core] if core in govs else '???'
        print(f"{core:<3}   {gov:<12}")


def set_governors(gov, rg = None):
    if gov not in _available_govs:
        print("ERROR: Not a valid governor. Please, try one of the following:")
        for av_gov in _available_govs:
            print(av_gov)
    else:
        if isinstance(rg, int):
            rg = [rg]
        elif rg is None:
            rg = _available_cpus

        for core in rg:
            try:
                cpufreq.cpuFreq().set_governors(gov, rg)
                print(f"CPU {core} set to {gov}.")
            except:
                print(f"ERROR: An exception occurred. Check if CPU {core} exists.")


def get_online():
    print("Online CPUs:" + "".join( f" {cpu}" for cpu in _available_cpus))


def set_online(rg = None):
    if isinstance(rg, int):
        rg = [rg]

    if rg is None:
        cpufreq.cpuFreq().enable_all_cpu()
        print("All CPUs enabled.")
    else:
        for core in rg:
            try:
                cpufreq.cpuFreq().enable_cpu(core)
                print(f"CPU {core} set online.")
            except:
                print(f"ERROR: An exception occurred. Check if CPU {core} exists.")


def set_offline(rg = None):
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _available_cpus

    for core in rg:
        try:
            cpufreq.cpuFreq().disable_cpu(core)
            print(f"CPU {core} set offline.")
        except:
            print(f"ERROR: An exception occurred. Check if CPU {core} exists.")


def set_minimum(freq, rg = None):
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _available_cpus

    for core in rg:
        try:
            cpufreq.cpuFreq().set_min_frequencies(freq, core)
            print(f"CPU {core} minimum frequency set to {int(freq/1000)} MHz.")
        except Exception as e:
            print(f"ERROR: An exception occurred for CPU {core}.")
            print(e)


def set_maximum(freq, rg = None):
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _available_cpus

    for core in rg:
        try:
            cpufreq.cpuFreq().set_max_frequencies(freq, core)
            print(f"CPU {core} maximum frequency set to {int(freq/1000)} MHz.")
        except Exception as e:
            print(f"ERROR: An exception occurred for CPU {core}.")
            print(e)


def set_frequency(freq, rg = None):
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _available_cpus

    for core in rg:
        try:
            minf = cpufreq.cpuFreq().get_min_freq()[core]
            if minf < freq:
                cpufreq.cpuFreq().set_max_frequencies(freq, core)
                cpufreq.cpuFreq().set_min_frequencies(freq, core)
            else:
                cpufreq.cpuFreq().set_min_frequencies(freq, core)
                cpufreq.cpuFreq().set_max_frequencies(freq, core)

            cpufreq.cpuFreq().set_frequencies(freq, core)
            print(f"CPU {core} frequency set to {int(freq/1000)} MHz.")
        except:
            print(f"ERROR: An exception occurred. Check if CPU {core} exists.")


def get_frequency(rg = None):
    if isinstance(rg, int):
        rg = [rg]
    elif rg is None:
        rg = _available_cpus

    freqs = cpufreq.cpuFreq().get_frequencies()
    minfs = cpufreq.cpuFreq().get_min_freq()
    maxfs = cpufreq.cpuFreq().get_max_freq()

    print("CPU   Current (MHz)   Minimum (MHz)   Maximum (MHz)")
    for core in rg:
        freq = int(freqs[core]/1000) if core in freqs else '???'
        minf = int(minfs[core]/1000) if core in minfs else '???'
        maxf = int(maxfs[core]/1000) if core in maxfs else '???'
        print(f"{core:<3}   {freq:<13}   {minf:<13}   {maxf:<13}")

##########################

def closest_frequency(freq):
    if freq < _available_freqs[0]:
        print("ERROR: Specified frequency is below the minimum allowed frequency.")
        exit()

    av_freq = _available_freqs[0]
    for af in _available_freqs:
        if freq <= av_freq:
            break
        av_freq = af

    return av_freq


def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    cpu_help = "CPU cores to watch or modify.\n"
    cpu_help += "All cores selected by default."
    parser.add_argument('-c', '--cpu', help=cpu_help, nargs='+', type=int, default=None)

    actions = parser.add_mutually_exclusive_group()
    
    gov_help = "'show' displays actual CPU governor schemes.\n"
    gov_help += "To change CPU governor scheme, use one of the following:\n"
    gov_help += "".join(f" {gov}" for gov in _available_govs)
    actions.add_argument('-g', '--governor', help=gov_help, default=argparse.SUPPRESS, type=str)

    online_help = "'show' displays online CPUs.\n"
    online_help += "'on' enables selected CPUs.\n"
    online_help += "'off' disables selected CPUs."
    actions.add_argument('-o', '--online', help=online_help, default=argparse.SUPPRESS, type=str)

    min_help = "sets CPUs minimum frequency to MIN (in MHz)."
    actions.add_argument('--min', help=min_help, default=argparse.SUPPRESS, type=int)

    max_help = "sets CPUs maximum frequency to MAX (in MHz)."
    actions.add_argument('--max', help=max_help, default=argparse.SUPPRESS, type=int)

    freq_help = "sets CPUs frequency to FREQ (in MHz)."
    actions.add_argument('-f', '--frequency', help=freq_help, default=argparse.SUPPRESS, type=int)

    show_help = "show CPUs frequency information."
    actions.add_argument('-s', '--show', help=show_help, action='store_true')



    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    rg = args.cpu

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

    if args.show:
        get_frequency(rg)


if __name__ == '__main__':
    main()
