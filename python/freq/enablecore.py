from cpufreq import cpuFreq

# CPU control setup.
cpu = cpuFreq()

# Enable CPU core.
core = int( input("Choose CPU: ") )
cpu.enable_cpu(core)
print(f"CPU {core} was set online.\n")
