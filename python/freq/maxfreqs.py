from cpufreq import cpuFreq

# CPU control setup
cpu = cpuFreq()

# Get and check max frequencies.
maxf = cpu.get_max_freq()
if not maxf:
    print("No maximum frequency reads available.")
    exit()

# Print max frecuencies by CPU.
print("CPU maximum frequencies (KHz):")
[ print(f"CPU {core}: {maxf[core]}") for core in sorted(maxf) ]