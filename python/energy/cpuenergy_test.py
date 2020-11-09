from cpufreq import cpuFreq
import pyRAPL
import time
import sys

# Handle script argv
args = len(sys.argv)
_interval = 1
_core = 0
_div = 8 # Socket size.
if args == 4:
    _interval = float(sys.argv[1])
    _core = int(sys.argv[2])
    _div = int(sys.argv[3])
elif args == 3:
    _interval = float(sys.argv[1])
    _core = int(sys.argv[2])
elif args == 2:
    _interval = int(sys.argv[1])

_socket = _core // _div

# Energy measurement setup.
pyRAPL.setup(devices = [pyRAPL.Device.PKG])
report = pyRAPL.outputs.PrintOutput()

# CPU control setup.
cpu = cpuFreq()
cpu.set_governors("userspace")

# Starting frequency: minimum
available_freqs = sorted(cpu.available_frequencies)
cpu.set_frequencies(available_freqs[0])
cpu.set_min_frequencies(available_freqs[0])
cpu.set_max_frequencies(available_freqs[0])

# Available frequencies energy test.
power_reg = {}
for freq in available_freqs[0:]:
    cpu.set_frequencies(freq)
    cpu.set_max_frequencies(freq)
    cpu.set_min_frequencies(freq)
    
    meter = pyRAPL.Measurement(f"{int(freq/1000)} MHz")
    
    meter.begin()
    time.sleep(_interval) # Measure power.
    meter.end()

    m_energy = meter._results.pkg[_socket] # micro-J
    m_time = meter._results.duration # micro-s
    m_power = m_energy / m_time # watts

    power_reg[freq] = m_power
    
    meter.export(report)
    print("\n")

print(f"{'Frequency (MHz)':<15}   {'Power (w)':<9}")
[ print(f"{int(freq/1000):<15}   {power_reg[freq]:.6f}") for freq in sorted(power_reg) ]