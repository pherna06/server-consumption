from cpufreq import cpuFreq
import sys

# Handle script argv
args = len(sys.argv)
core = None
if args == 2:
    core = int(sys.argv[1])

# Start CPU control
cpu = cpuFreq()

# Check core limits.
if core is not None and core not in cpu.get_online_cpus():
    print("ERROR: Selected core does not exist or is not online.")
    exit()

# Get all cores frequencies.
if core is None:
    freqs = cpu.get_frequencies()
    minfreqs = cpu.get_min_freq()
    maxfreqs = cpu.get_max_freq()
    print(f"{'CPU':<3}   {'Frequency (MHz)':<15}   {'Min':<4}   {'Max':<4}")
    [ print(f"{c:<3}   {int(freqs[c]/1000):<15}   {int(minfreqs[c]/1000):<4}   {int(maxfreqs[c]/1000):<4}") for c in sorted(freqs) ]
# Get selected core frequency.
else:
    freq = cpu.get_frequencies()[core]
    minfreq = cpu.get_min_freq()[core]
    maxfreq = cpu.get_max_freq()[core]
    print(f"{'CPU':<3}   {'Frequency (MHz)':<15}   {'Min':<4}   {'Max':<4}")
    print(f"{core:<3}   {int(freq/1000):<15}   {int(minfreq/1000):<4}   {int(maxfreq/1000):<4}")