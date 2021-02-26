import gym
import gymcpu
from gymcpu.envs.cpu_env00 import CPUEnv00
from gymcpu.envs.cpu_env01 import CPUEnv01
from gymcpu.envs.cpu_env02 import CPUEnv02

import numpypower as npp

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

import argparse
import pyRAPL
import os
import cpufreq
import shutil
import signal
import json

ENVIRONMENTS = {
    'CPUEnv00-v0':      CPUEnv00,
    'CPUEnv01-v0':      CPUEnv01,
    'CPUEnv02-v0':      CPUEnv02
}

# MODIFY ACCORDING TO YOUR MACHINE CPU CONFIGURATION.
CPU_LIST = list(range(16))
SOCKET_LIST = [0, 1]
SOCKET_DICT = {
        0: CPU_LIST[0:8], 
        1: CPU_LIST[8:16]
}

# cpufreq variables.
AVAILABLE_FREQS = sorted(cpufreq.cpuFreq().available_frequencies)

# ENVIRONMENT CONFIG
DEF_ENVCONFIG = {
    'socket' : 1,
    'cores'  : [8,9,10,11,12,13,14,15]
}

DEF_WORKCONFIG = {
    'size'   : 1000
}

DEF_TRAINCONFIG = {
    'epochs' : 5,
    'repeat' : 10
}

DEF_FILECONFIG = {
    'chkptpath' : 'temp/exa'
}

# MAX FREQUENCY
NOMINAL_MAXFREQ = 2601000
REAL_MAXFREQ = 3000000

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
    rg = []
    for skt in sockets:
        rg.extend(SOCKET_DICT[skt])

    return rg

def power_fork(work, group, size):
    """
        Handles forked processes for operation energy measurement. The affinity
        of the process is set to the given cores. The forked process will not 
        return; it has to be killed by the parent process.

        Parameters
        ----------
        work : str
            Name that specifies the operation to be measured.
        group : iterable(int)
            CPU cores to set process affinity.
        size : int
            Dimension of the Numpy elements used in the operation.
    """
    # Set core affinity.
    os.sched_setaffinity(0, group)

    # Start work operation:
    npp.powerop(op=work, config={'size': size})

def get_default_groups(envcores):
    return [[core] for core in envcores]

def get_powerpoints(args):
    minpower = args.envconfig.get('minpow', DEF_MINPOWER)
    maxpower = args.envconfig.get('maxpow', DEF_MAXPOWER)
    powers = []
    if 'powlist' in args:
        powers = args.powlist
    if 'powstep' in args: # Initial point: DEF_MINPOWER
        ppoint = minpower
        while ppoint <= maxpower:
            powers.append(ppoint)
            ppoint += args.powstep
    if 'pownum' in args: # Extreme points not used
        pstep = (maxpower - minpower) / (args.pownum + 1)
        ppoint = minpower + pstep
        for _ in range(args.pownum):
            powers.append(ppoint)
            ppoint += pstep

    return powers

#########################
#######################

def set_workload(work, config):
    size = config['size']
    pidls = []
    for group in config['groups']:
        pidls.append( os.fork() )
        if pidls[-1] == 0:
            power_fork(work, group, size)
            # Unreachable

    return pidls


def agent_learn(agent, chkpt, epochs):
    shutil.rmtree(chkpt, ignore_errors=True, onerror=None)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    for i in range(epochs):
        result = agent.train()
        chkpt_file = agent.save(chkpt)

        print( status.format(
            i + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
        ))

    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    return chkpt_file

def agent_measure(cpuenv, agent, measures):
    results = {}
    for _ in range(measures):
        state = cpuenv.reset()
        for _ in range(cpuenv.MAXSTEPS):
            action = agent.compute_action(state)
            state, _, done, info = cpuenv.step(action)

            if done:
                break
                
        freq = info['frequency']
        if freq in results:
            results[freq] += 1
        else:
            results[freq] = 1

    return results

def generate_logfiles(results, logpath):
    train_logpath = logpath + work + '.train.log'
    train_csvpath = logpath + work + '.train.csv'

    if os.path.exists(train_logpath):
        os.remove(train_logpath)
    if os.path.exists(train_csvpath):
        os.remove(train_csvpath)

    train_logf = open(train_logpath, 'w')
    train_csvf = open(train_csvpath, 'w')

    for powerpoint in powers:
        train_logf.write(f"Powerpoint: {powerpoint:.3f} w\n")
        train_logf.write("-------------------------\n") # 25-
        train_logf.write("Frequency (MHz)   Times\n")

        train_csvf.write(f"{powerpoint}\n")

        for freq in AVAILABLE_FREQS:
            if freq in results[powerpoint]:
                ftimes = results[powerpoint][freq]
            else:
                ftimes = 0

            train_csvf.write(f"{freq}, {ftimes}\n")

            train_logf.write(f"{freq//1000:<15}   {ftimes:<5}   |")
            [train_logf.write("·") for _ in range(ftimes)]
            train_logf.write("\n")

        train_logf.write("#########################\n\n") # 25#

    train_logf.close()
    train_csvf.close()

def learn_and_measure(
    env, envconfig, 
    work, workconfig, 
    powerpoints, agentconfig, trainconfig,
    fileconfig
    ):
    """
    """
    ### BACKGROUND WORKLOAD INITIALIZATION
    if 'groups' not in workconfig:
        workconfig['groups'] = get_default_groups(envconfig['cores'])

    workers = set_workload(work, workconfig)
    
    ## TRAINING
    ray.init(ignore_reinit_error=True)

    Env = ENVIRONMENTS[env]
    register_env(env, lambda config: Env(**config))

    results = {}
    for powerpoint in powers:
        # AGENT
        if agentconfig = None:
            config = ppo.DEFAULT_CONFIG.copy()
            
            config["log_level"]   = "WARN"
            config["num_workers"] = 0
        else:
            config = agentconfig

        envconfig['power']   = powerpoint
        config["env_config"] = envconfig

        agent = ppo.PPOTrainer(config, env=env)

        # TRAIN
        chkpt_file = agent_learn(
            agent  = agent,
            chkpt  = fileconfig['checkpoint'], 
            epochs = trainconfig['epochs']
        )

        # MEASURE
        cpuenv = gym.make(env, **config['env_config'])
        agent.restore(chkpt_file)
        
        point_results = agent_measure(
            cpuenv   = cpuenv,
            agent    = agent,
            measures = trainconfig['repeat']
        )

        results[powerpoint] = point_results
        print(point_results)

        # SAVE
        if 'save' in fileconfig:
            train_savepath = fileconfig['save'] + work + '.' + chkpt_file
            shutil.copy(chkpt_file, train_savepath)

    shutil.rmtree(fileconfig['checkpoint'], ignore_errors=True, onerror=None)

    ## BACKGROUND WORKLOAD KILL
    # Wait for all forked processes
    for pid in workers:
        os.kill(pid, signal.SIGKILL)

    ## RESULTS
    if 'log' in fileconfig:
        generate_logfiles(results, fileconfig['log'])
    

#########################
### COMMAND INTERFACE ###
#########################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    ## Dict for environment configuration.
    envconfig_help = "Dict of values for the configuration of the environment "
    envconfig_help += "in which the agent will be trained."
    parser.add_argument(
        '-e', '--envconfig'. metavar='envconfig', help=envconfig_help,
        type=json.loads,
        default=DEF_ENVCONFIG
    )

    ## Dict for workload configuration.
    workconfig_help = "Dict of values for the configuration of the background "
    workconfig_help += "worload."
    parser.add_argument(
        '-w', '--workconfig', metavar='workconfig', help=workconfig_help, 
        type=json.loads, 
        default=DEF_WORKCONFIG
    )

    ## Dict for agent configuration.
    agentconfig_help = "Dict of values for the configuration of the Ray agent."
    parser.add_argument(
        '-a', '--agentconfig', metavar='agentconfig', help=agentconfig_help,
        type=json.loads,
        default=None
    )

    ## Dict for train configuration.
    trainconfig_help = "Dict of values for the configuration of agent train."
    parser.add_argument(
        '-t', '--trainconfig', metavar='trainconfig', help=trainconfig_help,
        type=json.loads,
        default=DEF_TRAINCONFIG
    )

    ## Dict for files configuration.
    fileconfig_help = "Dict of filepaths used during training and results log."
    parser.add_argument(
        '-f', '--fileconfig', metavar='fileconfig', help=fileconfig_help,
        type=json.loads,
        default=DEF_FILECONFIG
    )

    ## Power steps for agents training.
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
    powernum_help += "bandwidth will be divided."
    power.add_argument(
        '--pownum', metavar='pownum', help=powernum_help,
        nargs=1,
        type=int,
        default=argparse.SUPPRESS
    )

    ## Measure process affinity: cores or sockets
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

    #log_help = "Path to folder where log files will be generated. "
    #log_help += "If not specified, log files will not be produced"
    #chkpt_help = "Path to folder where checkpoint files will be generated "
    #chkpt_help += "during agents training. Set by default to 'temp/exa'"
    #save_help = "Path to folder where trained agent files will be saved."
    #save_help += "If not specified, agent files will not be saved."

    # Positional arguments.
    work_help = "Name of operation to be tested: intproduct inttranspose "
    work_help += "intsort intscalar floatproduct floattranspose floatsort "
    work_help += "floatscalar"
    parser.add_argument('work', help=work_help)

    env_help = "Name of GYM environment that agents will use to be trained: "
    env_help += "CPUEnv00-v0 CPUEnv01-v0 CPUEnv02-v0."
    parser.add_argument('env', help=env_help)

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Socket must be selected for pyRAPL measurement.
    try:
        socket = args.envconfig['socket']
        pyRAPL.setup(
            devices = [pyRAPL.Device.PKG],
            socket_ids = [socket]
        )
    except:
        print("ERROR: check if selected sockets exist.")
        exit()

    # Measure process affinity
    if 'affcores' in args:
        os.sched_setaffinity(0, args.affcores)
    elif 'affsockets' in args:
        os.sched_setaffinity(0, get_cores(args.affsockets))

    if 'log' not in args.fileconfig:
        print(
            "WARNING: results will not be saved in log files.",
            "If needed, specify a log path with '-l' for results recording."
        )

    if 'save' not in args.fileconfig:
        print(
            "WARNING: trained agents configuration will not be saved.",
            "If needed, specify a folder path with '--save' to save them."
        )

    # Get power points
    powerpoints = get_powerpoints(args)

    learn_and_measure(
        env=args.env,
        envconfig=args.envconfig,
        work=args.work, 
        workconfig=args.workconfig, 
        powerpoints=powerpoints,
        agentconfig=args.agentconfig,
        trainconfig=args.trainconfig,
        fileconfig=args.fileconfig
    )


if __name__ == '__main__':
    main()
