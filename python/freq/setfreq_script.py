from cpufreq import cpuFreq
import sys

def set_core_limitfreq(freq, core):
    if cpu.get_min_freq()[core] < freq:
        cpu.set_max_frequencies(freq, core)
        cpu.set_min_frequencies(freq, core)
    else:
        cpu.set_min_frequencies(freq, core)
        cpu.set_max_frequencies(freq, core)


# Handle script argv
args = len(sys.argv)
freq = None
core = None
if args == 3:
    freq = int(sys.argv[1]) * 1000
    core = int(sys.argv[2])
elif args == 2:
    freq = int(sys.argv[1]) * 1000
else:
    print("ERROR: At least one argument (frequency) needed.")
    exit()

# Start CPU control
cpu = cpuFreq()
cpu.disable_hyperthread()

# Check frequency and core limits.
available_freqs = sorted(cpu.available_frequencies)
if freq < available_freqs[0]:
    print("ERROR: Frequency is less than minimum allowed frequency.")
    exit()

if core is not None and core not in cpu.get_online_cpus():
    print("ERROR: Selected core does not exist or is not online.")
    exit()

# Choose closest '<=freq' available frequency.
av_freq = available_freqs[0]
for af in available_freqs:
    if freq <= av_freq:
        break
    av_freq = af

# Set all cores to frequency.
if core is None:
    cpu.set_frequencies(av_freq)
    for c in cpu.get_online_cpus():
        set_core_limitfreq(av_freq, c)

    print(f"CPU frequency set to {int(av_freq/1000)} MHz.")
# Set selected core to frequency.        
else:
    cpu.set_frequencies(av_freq, core)
    set_core_limitfreq(av_freq, core)
    
    print(f"CPU {core} frequency set to {int(av_freq/1000)} MHz.")