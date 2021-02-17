import gym
import gymcpu
from gymcpu.envs.cpu_env00 import CPUEnv00
from gymcpu.envs.cpu_env01 import CPUEnv01
from gymcpu.envs.cpu_env02 import CPUEnv02

from gym_envs.gymtrain import ENVIRONMENTS
import power.numpypower as npp

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

import argparse
import pyRAPL
import os
import cpufreq
import shutil



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

# ENERGY MEASURE
DEF_POWERTIME = 2.0

# MAX FREQUENCY
NOMINAL_MAXFREQ = 2601000
REAL_MAXFREQ = 3000000

# DIMENSION
DEF_DIM = 1000

# CHECKPOINT ROOT PATH
DEF_CHKPTROOT = 'temp/exa'

# POWER BANDWIDTH
DEF_MINPOWER = 15.0
DEF_MAXPOWER = 115.0

# TRAINING EPOCHS
DEF_EPOCHS = 5

# REPEAT
DEF_REPEAT = 10

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

def power_fork(work, core, size):
    """
        Handles forked processes for operation energy measurement. The affinity
        of the process is set to the given core. The forked process will not 
        return; it has to be killed by the parent process.

        Parameters
        ----------
        work : str
            Name that specifies the operation to be measured.
        core : int
            CPU core to set process affinity.
        size : int
            Dimension of the Numpy elements used in the operation.
    """
    # Set core affinity.
    os.sched_setaffinity(0, {core})

    # Start work operation:
    npp.powerop(op=work, config={'size': size})

#########################
#######################

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
                freq = info['frequency']
                if freq in results:
                    results[freq] += 1
                else:
                    results[freq] = 0

    return results

    

def learn_and_measure(
    env, work, dim, powers, powertime, epochs, repeat,
    sockets, logpath, chkptpath, savepath
):
    """
    """
    ## WORKLOAD
    pidls = []
    for core in get_cores(sockets):
        pidls.append( os.fork() )
        if pidls[-1] == 0:
            power_fork(work, core, dim)
            # Unreachable
    
    ## TRAINING
    ray.init(ignore_reinit_error=True)

    Env = ENVIRONMENTS[env]
    register_env(env, lambda env_config: Env(**env_config))

    results = {}
    for powerpoint in powers:
        # AGENT
        env_config = {
            'socket': sockets,
            'cores': get_cores(sockets),
            'limit': powerpoint,
            'time': powertime
        }

        config = ppo.DEFAULT_CONFIG.copy()
        config["log_level"] = "WARN"
        config["num_workers"] = 0
        config["env_config"] = env_config
        agent = ppo.PPOTrainer(config, env=env)

        # TRAIN
        chkpt_file = agent_learn(
            agent=agent,
            chkpt=chkptpath, 
            epochs=epochs
        )

        # MEASURE
        cpuenv = gym.make(env, **config['env_config'])
        agent.restore(chkpt_file)
        
        point_results = agent_measure(
            cpuenv=env,
            agent=agent,
            measures=repeat
        )

        results[powerpoint] = point_results

        # SAVE
        if savepath is not None:
            train_savepath = savepath + work + '.' + chkpt_file
            shutil.copy(chkpt_file, train_savepath)

    ## END WORKLOAD
    # Wait for all forked processes
    for pid in pidls:
        os.waitpid(pid, 0)

    ## RESULTS
    train_logpath = logpath + work + '.train.log'
    train_csvpath = logpath + work + '.train.csv'

    if os.path.exists(train_logpath):
        os.remove(train_logpath)
    if os.path.exists(train_csvpath):
        os.remove(train_csvpath)

    train_logf = open(train_logpath, 'w')
    train_csvf = open(train_csvpath, 'w')

    for powerpoint in powers:
        train_logf.write("Powerpoint: {powerpoint:.3f} w\n")
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
            [train_logf.write("·" for _ in range(ftimes))]
            train_logf.write("\n")

        train_logf.write("#########################\n\n") # 25#

    train_logf.close()
    train_csvf.close()

#########################
### COMMAND INTERFACE ###
#########################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    ## Power steps for agents training.
    power = parser.add_mutually_exclusive_group(required=True)

    powerlist_help = "List of power 'points' for which agents will be trained "
    powerlist_help += "and results measured."
    power.add_argument(
        '--powlist', metavar='powlist', help=powerlist_help,
        nargs='+',
        type=float,
    )

    powerstep_help = "Step value to partition the power bandwidth into the "
    powerstep_help += "'points' for which agents will be trained and results "
    powerstep_help += "measured. The power bandwidth is "
    powerstep_help += "{}-{} watts.".format(DEF_MINPOWER, DEF_MAXPOWER)
    power.add_argument(
        '--powstep', metavar='powstep', help=powerstep_help,
        nargs=1,
        type=float
    )

    powerpoints_help = "Number of power 'points', for which agents will be "
    powerpoints_help += "trained and results measured, in which the power "
    powerpoints_help += "bandwidth will be divided."
    power.add_argument(
        '--powpoints', metavar='powpoints', help=powerpoints_help,
        nargs=1,
        type=int
    )

    ## Execution: socket.
    sockets_help = "The operation is executed in the cores of the specified "
    sockets_help += "sockets."
    parser.add_argument(
        '-s', '--sockets', metavar='sockets', help=sockets_help,
        nargs='+',
        type=int,
        required=True
    )

    ## Measure process affinity: cores or sockets
    affinity = parser.add_mutually_exclusive_group()

    affcores_help = "The cores in which the agent will be trained."
    affinity.add_argument(
        '--affcores', metavar='affcores', help=affcores_help,
        nargs='+',
        type=int
    )

    affsockets_help = "The sockets in whose cores the agent will be trained."
    affinity.add_argument(
        '--affsockets', metavar='affsockets', help=affsockets_help,
        nargs='+',
        type=int
    )

    # Optional arguments
    ## Dimension
    dim_help = "Dimension used for the Numpy elements in the operation. "
    dim_help += "For matrices, size is DIM X DIM."
    dim_help += "Default value for DIM is {}".format(DEF_DIM)
    parser.add_argument(
        '-d', '--dim', metavar='dim', help=dim_help, 
        type=int, 
        default=DEF_DIM
    )

    ## Energy measure time
    powertime_help = "Time (in seconds) to be spent in measuring energy "
    powertime_help += "consumption for each frequency"
    powertime_help += "Default value is {}".format(DEF_POWERTIME)
    parser.add_argument(
        '-t', '--powertime', metavar='powertime', help=powertime_help,
        type=float,
        default=DEF_POWERTIME
    )

    ## Training epochs.
    epochs_help = "Number of epochs for an agent to finish its training. "
    epochs_help += "Default value is {}.".format(DEF_EPOCHS)
    parser.add_argument(
        '-e', '--epochs', metavar='epochs', help=epochs_help,
        nargs=1,
        type=int,
        default=DEF_EPOCHS
    )

    ## Measure repeats.
    repeat_help = "Number of frequency measurements obtained from an agent "
    repeat_help += "for each power point."
    parser.add_argument(
        '-r', '--repeat', metavar='repeat', help=repeat_help,
        nargs=1,
        type=int,
        default=DEF_REPEAT
    )

    ## Log files path
    log_help = "Path to folder where log files will be generated. "
    log_help += "If not specified, log files will not be produced"
    parser.add_argument(
        '-l', '--log', metavar='log', help=log_help,
        nargs='?',
        type=str,
        default=None
    )

    ## Checkpoint path
    chkpt_help = "Path to folder where checkpoint files will be generated "
    chkpt_help += "during agents training. Set by default to 'temp/exa'"
    parser.add_argument(
        '--chkpt', metavar='chkpt', help=chkpt_help,
        nargs='?',
        type=str,
        default=DEF_CHKPTROOT
    )

    ## Model saves path
    save_help = "Path to folder where trained agent files will be saved."
    save_help += "If not specified, agent files will not be saved."
    parser.add_argument(
        '--save', metavar='save', help=save_help,
        nargs='?',
        type=str,
        default=None
    )

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
        pyRAPL.setup(
            devices = [pyRAPL.Device.PKG],
            socket_ids = args.sockets
        )
    except:
        print("ERROR: check if selected sockets exist.")
        exit()

    # Measure process affinity
    if 'affcores' in args:
        os.sched_setaffinity(0, args.affcores)
    elif 'affsockets' in args:
        os.sched_setaffinity(0, get_cores(args.affsockets))

    if args.log is None:
        print(
            "WARNING: results will not be saved in log files.",
            "If needed, specify a log path with '-l' for results recording."
        )

    if args.save is None:
        print(
            "WARNING: trained agents configuration will not be saved.",
            "If needed, specify a folder path with '--save' to save them."
        )

    # Get power points
    powers = []
    if 'powlist' in args:
        powers = args.powers
    if 'powstep' in args: # Initial point: DEF_MINPOWER
        ppoint = DEF_MINPOWER
        while ppoint <= DEF_MAXPOWER:
            powers.append(ppoint)
            ppoint += args.powstep
    if 'powpoints' in args: # Extreme points not used
        pstep = (DEF_MAXPOWER - DEF_MINPOWER) / (args.powpoints + 1)
        ppoint = DEF_MINPOWER + pstep
        for _ in range(args.powpoints):
            powers.append(ppoint)
            ppoint += pstep

    learn_and_measure(
        env=args.env, 
        work=args.work, 
        dim=args.dim, 
        powers=powers, 
        powertime=args.powertime,
        epochs=args.epochs,
        repeat=args.repeat,
        sockets=args.sockets, 
        logpath=args.log, 
        chkptpath=args.chkpt, 
        savepath=args.save
    )


if __name__ == '__main__':
    main()