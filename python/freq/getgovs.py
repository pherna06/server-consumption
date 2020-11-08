from cpufreq import cpuFreq

# CPU control setup
cpu = cpuFreq()

# Get and check governors.
govs = cpu.get_governors()
if not govs:
    print("No governor reads available.")
    exit()

# Print governor by CPU.
print("CPU governor schemes:")
[ print(f"CPU {core}: {govs[core]}") for core in sorted(govs) ]