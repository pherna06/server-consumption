import pyRAPL
import time

pyRAPL.setup(devices=[pyRAPL.Device.PKG])

report = pyRAPL.outputs.PrintOutput()

meter = pyRAPL.Measurement('bar', output = report)

meter.begin()
# Instructions to be evaluated
#
time.sleep(1)
#
#
meter.end()

meter.export(report)