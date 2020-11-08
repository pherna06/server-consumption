from cpufreq import cpuFreq
import pyRAPL
import time

def work():
    # Sleep for 1 second.
    time.sleep(5)


# Energy measurement setup.
pyRAPL.setup(devices = [pyRAPL.Device.PKG])
report = pyRAPL.outputs.PrintOutput()

# CPU control setup.
cpu = cpuFreq()
cpu.set_governors("userspace")

# Available frequencies energy test.
available_freqs = cpu.available_frequencies
for freq in available_freqs[1:]:
    cpu.set_frequencies(freq)
    
    meter = pyRAPL.Measurement(f"{int(freq/1000)} MHz")
    
    meter.begin()
    work()
    meter.end()
    
    meter.export(report)
    print("\n")