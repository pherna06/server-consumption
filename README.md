# Server Consumption

This repository contains the solutions for my end-of-degree project at 
*Universidad Complutense de Madrid*. The main goal of this project is the
development of a tool capable of optimizing energy consumption in a server
through the use of DVFS techniques, while avoiding drawbacks in the
performance of running processes.

*Read this in other languages: [English](README.md), [Español](README.es.md).*

## Progression

### *cpufreq* usage

Frequency scaling will be implemented by means of the
[cpufreq](https://github.com/VitorRamos/cpufreq) Python module.

On handling direct CPU frequency manipulation, I developed a command-line
interface, [cpufreq_utils.py](/cpufreq/cpufreq_utils.py), which makes use of
the utilities of *cpufreq* module.

#### Handling governor schemes

To know the current governor schemes in use by the CPU, we use:

```
cpufreq_utils.py -g show           // Show all cores schemes
cpufreq_utils.py -g show -c 2 3 4  // Show just schemes of cores 2, 3 and 4
```

To replace the current schemes for other, we just use the scheme name:
* *conservative*
* *ondemand*
* *userspace*
* *powersave*
* *performance*
* *schedutil*

```
cpufreq_utils.py -g userpsace          // All cores set to userspace
cpufreq_utils.py -g ondemand -c 2 3 4  // Just cores 2, 3 and 4 set to ondemand
```

#### Handling cores online state

CPU cores can be set *offline* so that they are not used in processing.
By default, all cores should be *online*.

To know which cores are online, we use:

```
cpufreq_utils.py -o show
```

To modify the cores online state, we use:

```
cpufreq_utils.py -o on           // All cores set online
cpufreq_utils.py -o on -c 2 3 4  // Just cores 2, 3 and 4 set online
```

```
cpufreq_utils.py -o off           // All cores set offline (WARNING)
cpufreq_utils.py -o off -c 2 3 4  // Just cores 2, 3 and 4 set offline
```

*On setting all cores offline, probably the system will prevent 
any dangers by keeping core 0 online.*

#### Checking frequencies

The cores' current frequencies, along with the minimum and maximum limits, can
be retrieved with:

```
cpufreq_utils.py -s           // Show all cores
cpufreq_utils.py -s -c 2 3 4  // Show just cores 2, 3 and 4
```

#### Setting frequencies

*\* Make sure that governor schemes are set to **userspace** when altering
frequencies. Otherwise the change will be ineffective.*

*\*\* Frequency modification might not reflect directly, as the cores will work
with lower frequencies than set if no workload is given to the processor."

The straightforward method to modify the cores' frequencies is using:

```
cpufreq_utils.py -f 1800           // Sets all cores to 1800 MHz
cpufreq_utils.py -f 1800 -c 2 3 4  // Sets cores 2, 3 and 4 to 1800 Mhz
```

With this method, not only the cores are set to the specified frequency, but
minimum and maximum limits are set to that frequency too. This is done to
ensure that the cores real frequency will be the picked one, as some CPUs
will just respond to the modification of minimum and maximum limits.

Limits can also be modified separately with the next commands:

```
cpufreq_utils.py --min 1200           // Set all cores minimum freq to 1200 MHz
cpufreq_utils.py --min 1200 -c 2 3 4  // Set cores 2, 3 and 4 minimum freq to 1200 MHz
cpufreq_utils.py --max 2100           // Set all cores maximum freq to 2100 MHz
cpufreq_utils.py --max 2100 -c 2 3 4  // Set cores 2, 3 and 4 maximum freq to 2100 MHz
```

Current frequency can be changed without altering minimum and maximum limits with:

```
cpufreq_utils.py --nolimits 1500           // Set all cores to 1500 MHz
cpufreq_utils.py --nolimits 1500 -c 2 3 4  // Set cores 2, 3 and 4 to 1500 MHz
```

#### Available governors and frequencies

As an assistance for CPU manipulation, we provide a command to display the
available frequency steps as well as the governor schemes.

```
cpufreq_utils.py -l freq  // Displays a list with available frequencies
cpufreq_utils.py -l gov   // Displays a list with governor schemes
```

#### Reset

CPU can be returned to its default configuration using:

```
cpufreq_utils.py -r
```

### Measuring execution time and energy consumption

A command-line interface, [numpymeasure](/measure/numpymeasure.py), is used to
measure execution time and power consumption of certain operations for 
particular frequencies from the [NumPy](https://github.com/numpy/numpy) module.

#### Execution time operations

Functions for handling the calculation of execution times for each operation
are implemented separately in [numpytime.py](/time/numpytime.py). 

This script has its own command for individually processing a particular
operation and obtaining its mean execution time directly in prompt.

```
numpytime floatproduct -c 2 3 4 -d 200 -r 1000
// Retrieves the mean execution time obtained from calculating the product of
// 200 x 200 real matrices, 1000 times, restringed to cores 2, 3 and 4.
```

```
numpytime floatproduct -s 1 -d 200 -r 1000
// Similar to previous command, but restringed to cores of socket no. 1.
```

#### Energy consumption operations

Functions for handling the calculation of power consumptions for each operation
are implemented separately in [numpypower.py](/power/numpypower.py).

This script has its own command too, but as opposed to *numpytime.py*, no
measurements are obtained. It is used to launch persistent processes that will
perform a particular operation in the assigned cores until they are killed
externally.

```
numpypower intsort -c 2 3 4 -d 10000
// Executes a process, assigned to cores 2, 3 and 4, that performs the sorting
// of a 10000 integers array repeatedly.
```

```
numpypower intsort -s 1 -d 10000
// Similar to previous command, but restringed to cores of socket no. 1.
```

*\* To facilitate external termination, the PID of the process is prompted.*

#### Measurement tool

The objective of calculating these metrics rely on doing so in function of the
CPU frequency. *numpymeasure.py* script provides a general purpose command-line
interface to do so.

```
numpymeasure floatproduct --time -c 2 3 4 -d 1000 -r 500 -l /measure/log/time/ -affcores 0 1
// Measures the mean execution time of calculating the product of 1000 x 1000 
// real matrices, repeated 500 times, for each core 2, 3 and 4.
// As no frequencies are specified, it will retrieve the time for each 
// available frequency. Log files will be generated in the provided folder
// path '/measure/log/time/'.
// The main process will be assigned to cores 0 and 1.
```

```
numpymeasure floatproduct --power -s 1 -f 1200 1600 2000 -d 1000 -t 0.5 --affsockets 0
// Measures the power consumption from repeatedly calculating the product of
// 1000 x 1000 real matrices for each core in socket no. 1, for frequencies
// 1200, 1600 and 300 MHz.
// Power consumption is measured at socket level and during 0.5 seconds.
// As no log path is specified, log files will not be generated.
// The main process wiil be assigned to cores in socket no. 0.
```

Requirements:
* Either *--time* or *--power* (but not both) must be specified, as the metric
  to be calculated.
* Either cores \[*-c*\] or sockets \[*-s*\] (but not both) must be specified.
  Particularly, when calculating power consumption, only sockets are valid.
* Either *--affcores* or *--affsockets* (but not both) can be specified, as the
  cores assigned to the process that handles measurement results. It should be
  assigned to cores other than the ones dedicated to the selected operation.
* Dimension \[*-d*\] value is optional; set by default to 1000.
* Repeat \[*-r*\] value is optional; set by default to 1. Used in *--time*.
* Powertime \[*-t*\] value is optional; set by default to 1.0. It indicates the
  time until energy variation (in Jules) is measured; power being calculated
  dividing energy between this time. Used in *--power*.

*\* For each operation, 2 log files will be generated: a **.csv** with raw data and
a more read-friendly **.log**.*

## Credits

## License

