import gym
import gymcpu
from   gymcpu.envs.cpu_env00 import CPUEnv00
from   gymcpu.envs.cpu_env01 import CPUEnv01
from   gymcpu.envs.cpu_env02 import CPUEnv02

import numpypower as npp

import argparse
import json

import os
import signal
import shutil





#########################
##### CONFIGURATION #####
#########################

DEF_CONFIG = {
    # POSITIONAL ARGUMENTS
    'env'  : '',
    'work' : '',
    # GYM CUSTOM ENVIRONMENTS
    'gymenvs' : {
        'CPUEnv00-v0' : CPUEnv00,
        'CPUEnv01-v0' : CPUEnv01,
        'CPUEnv02-v0' : CPUEnv02
    },
    # CPU DEFAULT SOCKET-CORES CONFIGURATION
    'cpuconfig' : {
        '0' : [0,1,2,3,4,5,6,7],
        '1' : [8,9,10,11,12,13,14,15]
    },
    # DEFAULT GYM ENVIRONMENT CONFIGURATION
    'envconfig' : {
        'socket' : 1,
        'cores'  : [8,9,10,11,12,13,14,15]
    },
    # DEFAULT WORKLOAD CONFIGURATION
    'workconfig' : {
        'size'   : 1000,
        'groups' : [[core] for core in [8,9,10,11,12,13,14,15]]
    },
    # DEFAULT TRAINING CONFIGURATION
    'trainconfig' : {
        'epochs'    : 5,
        'chkptpath' : 'trained_agents/default',
        'verbose'   : True
    }
}

def load_config(args):
    # DEFAULT CONFIG LOADED FIRST
    config = DEF_CONFIG

    # FIELDS MODIFIED WITH THOSE OF GENERAL CONFIG
    if 'config' in args:
        for field in args.config:
            config[field] = args.config[field]
    
    # FIELDS MODIFIED WITH THOSE OF PARTICULAR CONFIGS
    argsdict = vars(args)
    for field in config:
        if field in args:
            config[field] = argsdict[field]

def save_config(env, work, config):
    config['env']  = env
    config['work'] = work

    with open(config['chkptpath']) as jsonf:
        json.dump(config, jsonf)

#########################
# --------------------- #
#########################



    

#########################
### UTILITY FUNCTIONS ###
#########################

def read_json(jsonpath):
    with open(jsonpath) as jsonf:
        return json.load(jsonf)

def get_cores(sockets, config):
    """
        Retrieves the cores associated to the given socket IDs.

        Parameters
        ----------
        sockets : list
            The specified socket numbers.
        config : dict(str, list)
            List with the cores assigned to each socket.
        
        Returns
        -------
        list
            The cores associated with the given socket numbers.
    """
    cores = []
    for skt in sockets:
        cores.extend(config[str(skt)])

    return cores

#########################
# --------------------- #
#########################






#########################
## WORKLOAD FUNCTIONS ###
#########################

def fork_numpywork(work, group, size):
    """
        Performs a fork to create a subprocess where the Numpy-work task will 
        be executed. The affinity of the process is set to the given group of 
        cores. The forked process will not return; it has to be killed.

        Parameters
        ----------
        work : str
            Name that specifies the work operation to execute.
        group : iterable(int)
            CPU cores to set the process affinity.
        size : int
            Dimension of the Numpy elements used in the operation.

        Returns
        -------
        pid : int
            The PID of the forked subprocess.
    """
    pid = os.fork()

    if pid == 0:
        # Set core affinity.
        os.sched_setaffinity(0, group)

        # Start work operation:
        npp.powerop(op=work, config={'size': size})
        ## unreachable
    
    return pid

def start_work(work, config):
    """
        Starts the subprocesses where the work will take place.

        Parameters
        ----------
        work : str
            Name that specifies the work operation to execute.
        config : dict
            Configuration for the workload.

        Returns
        -------
        workers : list(int)
            List with the created subprocesses PIDs.
    """
    size = config['size']
    groups = config['groups']

    workers = []
    for group in groups:
        taskpid = fork_numpywork(work, group, size)
        workers.append(taskpid)

    return workers

def end_work(workers):
    """
        Kills all work subprocesses specified.

        Parameters
        ----------
        workers : list(int)
            List with the subprocesses PIDs.
    """
    for pid in workers:
        os.kill(pid, signal.SIGKILL)

#########################
# --------------------- #
#########################





#########################
## TRAINING FUNCTIONS ###
#########################

def set_PPOagent(env, envconfig, agentconfig):
    """
        Creates a PPO agent from a given GYM environment and its configuration
        and a custom agent configuration. If not specified, default PPO
        configuration is used.

        Parameters
        ----------
        env : str
            Name of the GYM environment to use for the agent.
        envconfig : dict
            Configuration of the GYM environment.
        agentconfig : dict
            Configuration of the RAY agent.

        Returns
        -------
        agent : PPOTrainer
            The generated agent.
    """
    import ray.rllib.agents.ppo as ppo

    config = agentconfig
    if config is None:
        config = ppo.DEFAULT_CONFIG.copy()

        config['log_level']   = 'WARN'
        config['num_workers'] = 0

    config['env_config'] = envconfig

    agent = ppo.PPOTrainer(config, env=env)
    return agent

def train_agent(agent, config):
    """
        Trains an already configured agent according to the training 
        configuration specified. Training checkpoints are stored on the
        specified path.

        Parameters
        ----------
        agent : PPOTrainer
            Agent to be trained.
        config : dict
            Configuration of the training.
    """
    epochs = config['epochs']
    chkpt = config['chkptpath']
    verbose = config['verbose']

    # Remove folder.
    shutil.rmtree(chkpt, ignore_errors=True, onerror=None)

    # Status string.
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

    # Train for X epochs.
    for i in range(epochs):
        result = agent.train()
        chkpt_file = agent.save(chkpt)

        if verbose:
            print( status.format(
                i + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
            ))

    # Show trained agent summary.
    if verbose:
        policy = agent.get_policy()
        model = policy.model
        print(model.base_model.summary())

def train(env, work, config):
    """
        Generates a PPO agent based on the specified GYM environment and agent
        configuration. The agent is trained while the specified workload runs
        in the background, and its checkpoints stored in the path indicated in
        the training configuration.

        Parameters
        ----------
        env : str
            Name of the GYM environment to use for the agent.
        work : str
            Name of the workload operation to be run in the background.
        config : dict
            General configuration of the training environment.
    """
    import ray
    from  ray.tune.registry import register_env

    ## REGISTER ENVIRONMENT
    Env = config['gymenvs'][env]
    register_env(env, lambda config: Env(**config))
    
    ## AGENT CONFIGURATION
    ray.init(ignore_reinit_error=True)
    agent = set_PPOagent(env, config['envconfig'], config['agentconfig'])

    ## BACKGROUND WORKLOAD INITIALIZATION
    workers = start_work(work, config['workconfig'])

    ## TRAINING
    train_agent(agent, config['trainconfig'])

    ## BACKGROUND WORKLOAD KILL
    end_work(workers)

    # SAVE TRAINING CONFIGURATION
    save_config(env, work, config)
    
#########################
# --------------------- #
#########################





#########################
### COMMAND INTERFACE ###
#########################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    ## GENERAL CONFIGURATION
    genconfig_help = "Dict with the general configuration for the script."
    parser.add_argument(
        '-g', '--config', metavar='config', help=genconfig_help,
        type=read_json,
        default=DEF_CONFIG
    )

    ## GYM ENVIRONMENT
    env_help = "Name of GYM environment that agents will use to be trained: "
    env_help += "CPUEnv00-v0 CPUEnv01-v0 CPUEnv02-v0."
    parser.add_argument('env', help=env_help)

    envconfig_help = "Dict of values for the configuration of the environment "
    envconfig_help += "in which the agent will be trained."
    parser.add_argument(
        '-e', '--envconfig', metavar='envconfig', help=envconfig_help,
        type=read_json,
        default=argparse.SUPPRESS
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
        type=read_json, 
        default=argparse.SUPPRESS
    )

    ## AGENT CONFIGURATION
    agentconfig_help = "Dict of values for the configuration of the Ray agent."
    parser.add_argument(
        '-a', '--agentconfig', metavar='agentconfig', help=agentconfig_help,
        type=read_json,
        default=argparse.SUPPRESS
    )

    ## TRAINING CONFIGURATION
    trainconfig_help = "Dict of values for the configuration of agent train."
    parser.add_argument(
        '-t', '--trainconfig', metavar='trainconfig', help=trainconfig_help,
        type=read_json,
        default=argparse.SUPPRESS
    )

    ## CPU CONFIGURATION
    cpuconfig_help = "Dict of socket-cores CPU relation."
    parser.add_argument(
        '-c', '--cpuconfig', metavar='cpuconfig', help=cpuconfig_help,
        type=read_json,
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

    # SET CONFIGURATION
    config = load_config(args)

    # SET POSITIONAL ARGS
    env = args.env
    if env == 'defenv':
        env = config['env']
    
    work = args.work
    if work == 'defwork':
        work = config['work']


    # SET TRAINING PROCESS AFFINITY
    if 'affcores' in args:
        os.sched_setaffinity(0, args.affcores)
    elif 'affsockets' in args:
        cores = get_cores(args.affsockets, config['cpuconfig'])
        os.sched_setaffinity(0, cores)

    ## TRAIN
    train(
        env    = env, 
        work   = work,
        config = config
    )


if __name__ == '__main__':
    main()
