from os import get_terminal_size
from cpufreq import cpuFreq
import time

# CPU control setup.
cpu = cpuFreq()
cpu.disable_hyperthread()

# Get and check online CPUs
online_cpus = cpu.get_online_cpus()
if not online_cpus:
    print("No online CPUs available.\n")
    exit()

# List available CPUs. Get valid CPU input.
print("Online CPUs:")
[ print(f"{core}") for core in online_cpus ]
print("-1 to select all cores")
core = int( input("Choose CPU: ") )

while core not in online_cpus + [-1]:
    core = int( input("Choose a valid CPU: ") )

if core == -1:
    core = None

# Get and check available frequencies.
available_freqs = cpu.available_frequencies
if not available_freqs:
    print("No frequencies available.\n")
    exit()

# List available frequencies. Get valid frequency.
print("Available frequencies:")
[ print(f"{i}: {int(freq/1000)} MHz") for (i, freq) in enumerate(available_freqs) ]
idx = int( input("Choose wanted frequency index: ") )

while idx not in range( len(available_freqs) ):
    idx = int( input("Choose a valid index: ") )

freq = available_freqs[idx]

# Set chosen frequency for chosen CPU.
cpu.set_governors("userspace", core)
cpu.set_frequencies(freq, rg = core)

if core is not None:
    if cpu.get_min_freq()[core] > freq:
        cpu.set_min_frequencies(freq, rg = core)
        cpu.set_max_frequencies(freq, rg = core)
    else:
        cpu.set_max_frequencies(freq, rg = core)
        cpu.set_min_frequencies(freq, rg = core)

    print(f"CPU {core} frequency limited to {int(freq/1000)} MHz.\n")

    time.sleep(1)
    curr = cpu.get_frequencies()[core]
    print(f"CPU {core} current frequency (in KHz): {curr}")
else:
    for c in cpu.get_online_cpus():
        if cpu.get_min_freq()[c] > freq:
            cpu.set_min_frequencies(freq, rg = c)
            cpu.set_max_frequencies(freq, rg = c)
        else:
            cpu.set_max_frequencies(freq, rg = c)
            cpu.set_min_frequencies(freq, rg = c)

    print(f"CPU frequency limited to {int(freq/1000)} MHz.\n")

    time.sleep(1)
    freqs = cpu.get_frequencies()
    print("CPU frequencies (MHz):")
    [ print(f"CPU {core}: {int(freqs[core]/1000)}") for core in sorted(freqs) ]