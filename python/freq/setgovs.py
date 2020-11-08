from cpufreq import cpuFreq

# CPU control setup
cpu = cpuFreq()

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

# List and get available govs
available_govs = cpu.available_governors
print("Available governor schemes:")
[ print(gov) for gov in available_govs ]
gov = input("Choose a governor scheme: ")

if not available_govs:
    print("Not governor schemes available.")
    exit()

while gov not in available_govs:
    gov = input("Choose a valid governor scheme: ")

# Set core to governor scheme
cpu.set_governors(gov, rg = core)
if core:
    print(f"CPU {core} scheme set to {gov}.\n")
else:
    print(f"CPU global scheme set to {gov}.\n")

if core:
    curr = cpu.get_governors()[core]
    print(f"CPU {core} current scheme is: {curr}")
else:
    govs = cpu.get_governors()
    print("CPU schemes:")
    [ print(f"CPU {core}: {govs[core]}") for core in sorted(govs) ]