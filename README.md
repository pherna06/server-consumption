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




## Credits

## License

