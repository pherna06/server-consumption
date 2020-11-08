from cpufreq import cpuFreq

# CPU control setup.
cpu = cpuFreq()

# Get and check online CPUs
online_cpus = cpu.get_online_cpus()
if not online_cpus:
    print("No online CPUs available.\n")
    exit()

# List available CPUs. Get valid CPU input.
print("Online CPUs:")
[ print(f"{core}") for core in online_cpus ]
core = int( input("Choose CPU: ") )

while core not in online_cpus:
    core = int( input("Choose a valid CPU: ") )

# Disable CPU core
cpu.disable_cpu(core)
print(f"CPU {core} was set offline.\n")