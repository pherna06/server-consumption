from cpufreq import cpuFreq

# CPU control setup
cpu = cpuFreq()

# Get and check current frequencies.
freqs = cpu.get_frequencies()
if not freqs:
    print("No frequency reads available.")
    exit()

# Print frecuencies by CPU.
print("CPU frequencies (KHz):")
[ print(f"CPU {core}: {freqs[core]}") for core in sorted(freqs) ]