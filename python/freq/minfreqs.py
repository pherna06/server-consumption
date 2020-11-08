from cpufreq import cpuFreq

# CPU control setup
cpu = cpuFreq()

# Get and check min frequencies.
minf = cpu.get_min_freq()
if not minf:
    print("No minimum frequency reads available.")
    exit()

# Print min frecuencies by CPU.
print("CPU minimum frequencies (KHz):")
[ print(f"CPU {core}: {minf[core]}") for core in sorted(minf) ]