import argparse
import pyRAPL
import os
import cpufreq
import shutil
import signal
import json

# CPU DEFAULT SOCKET-CORES CONFIGURATION
CORES_LIST  = list(range(16))
SOCKET_LIST = [0, 1]
SOCKET_DICT = {
    0: CORES_LIST[0:8], 
    1: CORES_LIST[8:16]
}

# ENVIRONMENT CONFIG
DEF_ENVCONFIG = {
    'socket' : 1,
    'cores'  : [8,9,10,11,12,13,14,15]
}

# DEFAULT WORKLOAD CONFIGURATION
DEF_WORKCONFIG = {
    'size'   : 1000,
    'groups' : [[core] for core in DEF_ENVCONFIG['cores']]
}

# DEFAULT TRAINING CONFIGURATION
DEF_TRAINCONFIG = {
    'epochs'    : 5,
    'chkptpath' : 'trained_agents/default',
    'verbose'   : True
}

# DEFAULT TEST CONFIGURATION
DEF_TESTCONFIG = {
    'logpath':   'tests/default',
    'iter' :     10,
    'verbose':   True
}

# POWER BANDWIDTH
DEF_MINPOWER = 15.0
DEF_MAXPOWER = 115.0





#####################
# UTILITY FUNCTIONS #
#####################

def get_cores(sockets):
    """
        Retrieves the cores associated to the given socket IDs.

        Parameters
        ----------
        sockets : list
            The specified socket numbers.
        
        Returns
        -------
        list
            The cores associated with the given socket numbers.
    """
    cores = []
    for skt in sockets:
        cores.extend(SOCKET_DICT[skt])

    return cores

def get_powerpoints(args):
    """
        Handles the parser arguments associated with powerpoints selection.
        If 'powlist', the specified list of powerpoints is returned. 
        If 'powstep', the points are calculated with the step from the default
        minimum powerpoint. 
        If 'pownum', the power bandwidth (from minimum to maximum points) is
        equispaced with the specified number of powerpoints.

        Parameters
        ----------
        args
            Arguments from script parser.

        Returns
        -------
        powerpoints : list(float)
            List with the desired powerpoints.
    """
    if 'powlist' in args:
        return args.powlist

    minpower = args.envconfig.get('minpow', DEF_MINPOWER)
    maxpower = args.envconfig.get('maxpow', DEF_MAXPOWER)
    powerpoints = []
    if 'powstep' in args: # Initial point: DEF_MINPOWER
        ppoint = minpower
        while ppoint <= maxpower:
            powerpoints.append(ppoint)
            ppoint += args.powstep
    if 'pownum' in args: # Extreme points not used
        pstep = (maxpower - minpower) / (args.pownum + 1)
        ppoint = minpower + pstep
        for _ in range(args.pownum):
            powerpoints.append(ppoint)
            ppoint += pstep

    return powerpoints

#########################
# --------------------- #
#########################





#########################
#### CSV GENERATION #####
#########################

def generate_csv(results, path):
    """
        Creates or overwrites a CSV file with the frequency mean and mode
        of the results for each powerpoint in the specified folder.

        Parameters
        ----------
        results : dict(float, int, int)
            Results of test, storing the frequency count for each powerpoint.
        csvpath : str
            Path of the folder where the CSV will be created.
    """
    csvpath = path + '/results.csv'

    if os.path.exists(csvpath):
        os.remove(csvpath)

    csvf = open(csvpath, 'w')

    # Header
    csvf.write("Powerpoint, Mean, Mode\n")

    # Results
    for power in results:
        mean  = 0
        count = 0
        mode  = 0
        maxc  = 0
        for freq in results[power]:
            freqc = results[power][freq]

            mean  += freq * freqc
            count += freqc

            if freqc > maxc:
                 mode = freq

        mean /= count

        csvf.write(f"{power}, {mean}, {mode}\n")

    csvf.close()

#########################
# --------------------- #
#########################





#########################
#### TRAIN AND TEST #####
#########################

def train_and_test(
    env, envconfig, 
    work, workconfig, 
    powerpoints, 
    agentconfig, 
    trainconfig, testconfig
    ):
    """
        Trains and tests PPO agents generated from the specified GYM
        environment for each indicated powerpoint. Results for each
        agent are stored as well as general results of frequency mean
        and mode for each powerpoint.

        Parameters
        ----------
        env : str
            Name of the GYM environment to test.
        envconfig : dict
            Configuration of the GYM environment.
        work : str
            Name of the workload operation to be run in the background.
        workconfig : dict
            Configuration of the background workload.
        powerpoints : list(float)
            List of 'power points' assigned to each agent environment as
            goal for training.
        agentconfig : dict
            Configuration of the PPO agent.
        trainconfig : dict
            Configuration of the training process.
        testconfig : dict
            Configuration of the test process.
    """
    from raytrain import train
    from raytest import test

    ## GENERAL PATHS
    trainpath = trainconfig['chkptpath']
    testpath  = testconfig['logpath']

    ## TRAIN-TEST PER POWERPOINT
    results = {}
    for i, power in enumerate(powerpoints):
        ## ADAPT TRAINING
        envconfig['power'] = power
        trainconfig['chkptpath'] = trainpath + f"/powerpoint-{i + 1}"

        ## TRAIN
        train(
            env, envconfig,
            work, workconfig,
            agentconfig, trainconfig
        )

        ## TRAINED PATH
        epochs = trainconfig['epochs']
        agentpath = trainconfig['chkptpath'] + f"/checkpoint-{epochs}"

        ## ADAPT TEST
        testconfig['logpath'] = testpath + f"/powerpoint-{i + 1}"

        ## TEST
        count, _ = test(
            agentpath,
            env, envconfig,
            work, workconfig,
            testconfig
        )

        ## RECORD COUNT
        results[power] = count

    ## RESULTS
    generate_csv(results, testpath)

#########################
# --------------------- #
#########################


        
    

#########################
### COMMAND INTERFACE ###
#########################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    ## GYM ENVIRONMENT
    env_help = "Name of GYM environment that agents will use to be trained: "
    env_help += "CPUEnv00-v0 CPUEnv01-v0 CPUEnv02-v0."
    parser.add_argument('env', help=env_help)

    envconfig_help = "Dict of values for the configuration of the environment "
    envconfig_help += "in which the agent will be trained."
    parser.add_argument(
        '-e', '--envconfig', metavar='envconfig', help=envconfig_help,
        type=json.loads,
        default=DEF_ENVCONFIG
    )

    ## WORKLOAD
    work_help = "Name of operation to be tested: intproduct inttranspose "
    work_help += "intsort intscalar floatproduct floattranspose floatsort "
    work_help += "floatscalar"
    parser.add_argument('work', help=work_help)

    workconfig_help = "Dict of values for the configuration of the background "
    workconfig_help += "worload."
    parser.add_argument(
        '-w', '--workconfig', metavar='workconfig', help=workconfig_help, 
        type=json.loads, 
        default=DEF_WORKCONFIG
    )

    ## AGENT CONFIGURATION
    agentconfig_help = "Dict of values for the configuration of the Ray agent."
    parser.add_argument(
        '-a', '--agentconfig', metavar='agentconfig', help=agentconfig_help,
        type=json.loads,
        default=None
    )

    ## TRAINING CONFIGURATION
    trainconfig_help = "Dict of values for the configuration of agent train."
    parser.add_argument(
        '-t', '--trainconfig', metavar='trainconfig', help=trainconfig_help,
        type=json.loads,
        default=DEF_TRAINCONFIG
    )

    ## TEST CONFIGURATION
    testconfig_help = "Dict of values for the configuration of agent testing."
    parser.add_argument(
        '-x', '--testconfig', metavar='testconfig', help=testconfig_help,
        type=json.loads,
        default=DEF_TESTCONFIG
    )

    ## POWERSTEPS
    power = parser.add_mutually_exclusive_group(required=True)

    powerlist_help = "List of power 'points' for which agents will be trained "
    powerlist_help += "and results measured."
    power.add_argument(
        '--powlist', metavar='powlist', help=powerlist_help,
        nargs='+',
        type=float,
        default=argparse.SUPPRESS
    )

    powerstep_help = "Step value to partition the power bandwidth into the "
    powerstep_help += "'points' for which agents will be trained and results "
    powerstep_help += "measured. The power bandwidth is "
    powerstep_help += "{}-{} watts.".format(DEF_MINPOWER, DEF_MAXPOWER)
    power.add_argument(
        '--powstep', metavar='powstep', help=powerstep_help,
        nargs=1,
        type=float,
        default=argparse.SUPPRESS
    )

    powernum_help = "Number of power 'points', for which agents will be "
    powernum_help += "trained and results measured, in which the power "
    powernum_help += "bandwidth will be divided. The power bandwidth is "
    powernum_help += "{}-{} watts.".format(DEF_MINPOWER, DEF_MAXPOWER)
    power.add_argument(
        '--pownum', metavar='pownum', help=powernum_help,
        nargs=1,
        type=int,
        default=argparse.SUPPRESS
    )

    ## PROCESS AFFINITY
    affinity = parser.add_mutually_exclusive_group()

    affcores_help = "The cores in which the agent will be trained."
    affinity.add_argument(
        '--affcores', metavar='affcores', help=affcores_help,
        nargs='+',
        type=int,
        default=argparse.SUPPRESS
    )

    affsockets_help = "The sockets in whose cores the agent will be trained."
    affinity.add_argument(
        '--affsockets', metavar='affsockets', help=affsockets_help,
        nargs='+',
        type=int,
        default=argparse.SUPPRESS
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # SET TEST PROCESS AFFINITY
    if 'affcores' in args:
        os.sched_setaffinity(0, args.affcores)
    elif 'affsockets' in args:
        os.sched_setaffinity(0, get_cores(args.affsockets))

    # Get power points
    powerpoints = get_powerpoints(args)

    train_and_test(
        env          = args.env,
        envconfig    = args.envconfig,
        work         = args.work, 
        workconfig   = args.workconfig, 
        powerpoints  = powerpoints,
        agentconfig  = args.agentconfig,
        trainconfig  = args.trainconfig
    )


if __name__ == '__main__':
    main()
