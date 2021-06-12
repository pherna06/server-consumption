# Local module that registers custom environments in GYM.
import gymcpu

# GYM functions to deal with those environments.
from gym import make     as gym_make
from gym import register as gym_register
from gym import spec     as gym_spec

# Function to obtain a callable from an environment entry_point.
from gym.envs.registration import load as load_env_class

import numpypower as npp

import argparse
import json

import os
import signal
import shutil

import matplotlib.pyplot as plt





#########################
##### CONFIGURATION #####
#########################

DEF_CONFIG = {
    # ENVIRONMENT AND WORK ID NAMES
    'env'  : '',
    'work' : '',
    # CPU DEFAULT SOCKET-CORES CONFIGURATION
    'cpuconfig' : {
        '0' : [0]
    },
    # DEFAULT GYM ENVIRONMENT CONFIGURATION
    'envconfig' : {
        'socket' : 0,
        'cores'  : [0]
    },
    # DEFAULT WORKLOAD CONFIGURATIONgit 
    'workconfig' : {
        'size'   : 1000,
        'groups' : [[0]]
    },
    # DEFAULT TEST CONFIGURATION
    'testconfig' : {
        'iter'      : 1,
        'chkptpath' : 'tests/default',
        'verbose'   : 1,

        'init_freqs' : []
    }
}

def load_config(args):
    # DEFAULT CONFIG LOADED FIRST
    config = {}
    if args.loadconfig:
        config = read_json( args.agent + '/config.json' )
    else:
        config = args.config

    # FIELDS MODIFIED WITH GENERAL AND PARTICULAR CONFIGURATIONS
    argsdict = vars(args)
    for field in DEF_CONFIG:
        if field in args:
            config[field] = argsdict[field]

    return config

def save_config(config, path):
    with open(path + '/config.json', 'w+') as jsonf:
        json.dump(config, jsonf, indent = 4)

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
# LOG & CSV GENERATION ##
#########################

def generate_log(history, path):
    """
        Given the status history log of an agent test, a CSV file is generated 
        with the results and a log file is generated with the status history of 
        the environment during the test iterations. Both files are generated in 
        the specified path.

        Parameters
        ----------
        history : dict(str, str, dict)
            Status history of the environment during test, for each iteration 
            and step.
        path : str
            Path where files will be created (overwritten if necessary)
        
    """
    ## CREATE FOLDER
    os.makedirs( os.path.dirname(path), exist_ok=True)

    ## STATUS HISTORY LOG
    histpath = path + '/test.log'
    history_log(history, histpath)

    ## STATUS HISTORY CSV
    history_csv(history, path)

def history_log(history, logpath):
    """
        Creates or overwrites a log file with the status history of the tested
        environment during the test iterations.

        Parameters
        ----------
        history : dict(str, str, dict)
            Status history of the environment during test, for each iteration 
            and step.
        logpath : str
            Namepath of the log file.
    """
    # Create or overwrite
    if os.path.exists(logpath):
        os.remove(logpath)
    logf = open(logpath, 'w+')

    for it in history:
        logf.write(f"{it}\n")
        logf.write("-------------------------\n") #25-

        for step in history[it]:
            logf.write(f"{step}\n")
            for data in history[it][step]:
                value = history[it][step][data]
                logf.write(f"\t{data}: {value}\n")

        logf.write("\n")
        logf.write("#########################\n\n") # 25#

    logf.close()

def history_csv(history, path):
    """
        Creates a CSV file for each iteration reflecting the environment status
        in each step.

        Parameters
        __________
        history : dict(str, str, dict)
            Status history of the environment during test, for each iteration
            and step.
        path : str
            Namepath of directory where CSV files will be generated.
    """
    iternum = 0
    for it in history:
        # CSV FILE CREATION
        csvpath = path + f'iter-{iternum}.csv'
        if os.path.exists(csvpath):
            os.remove(csvpath)
        csvf = open(csvpath, 'w+')
        iternum += 1

        # CSV HEADER
        csvf.write('Step')
        status = history[it][0]
        for key in status:
            if key == 'step':
                continue
            csvf.write(f',{key}')
        csvf.write('\n')

        # CSV CONTENT
        for step in history[it]:
            csvf.write(f'{step}')
            status = history[it][step]
            for key in status:
                if key == 'step':
                    continue
                csvf.write(f',{status[key]}')
            csvf.write('\n')

#########################
# --------------------- #
#########################





#########################
### DISPLAY FUNCTIONS ###
#########################

def display_status(label, status):
    """
        Displays, in standard output, the passed status of an environment,
        preceded by a label. Asks for input to continue the execution.

        Parameters
        ----------
        label : str
            Name assigned to this status.
        status : dict(str)
            Status of an environment variables.
    """
    print(label)
    for data in status:
        print(f"\t{data}: {status[data]}")

    input("Press enter to continue...")

#########################
# --------------------- #
#########################





#########################
#### TEST FUNCTIONS #####
#########################

def get_PPOagent(env, envconfig, chkptpath):
    """
        Retrieves an already trained PPO agent from the indicated path,
        based on the specified GYM environment.

        Parameters
        ----------
        env : str
            Name of the GYM environment to use for the agent.
        envconfig : dict
            Configuration of the GYM environment.
        chkptpath : str
            Path of folder where the trained agent checkpoint is stored.

        Returns
        -------
        agent : PPOTrainer
            The trained agent.
    """
    import ray.rllib.agents.ppo as ppo

    config = ppo.DEFAULT_CONFIG.copy()
    config['log_level']   = 'WARN'
    config['num_workers'] = 0
    config['env_config']  = envconfig

    agent = ppo.PPOTrainer(config, env=env)
    agent.restore(chkptpath)

    return agent

def test_env(testenv, agent, config):
    """
        Test of a GYM environment with an agent deciding on actions based
        on environment state. The test is repeated for the indicated 
        iterations. Status of environment in each step is displayed if
        verbose activated. Test results (frequency count) and status
        history are returned.

        Parameters
        ----------
        env : str
            Name of the GYM environment to test.
        envconfig : dict
            Configuration of the GYM environment.
        agent
            Trained agent that will decide over actions.
        iterations : int
            Number of times the environment test will be repeated.
        verbose : bool
            Wheter status of environment in each step is displayed in output.

        Returns
        -------
        history : dict(str, str, dict)
            Status history of the environment during test, for each iteration 
            and step.
    """
    iterations = config['iter']
    verbose    = config['verbose']

    init_freqs = config['init_freqs']
    numfreqs   = len(init_freqs)
    freqcount  = 0

    history = {}

    # TEST ITERATIONS
    for i in range(iterations):
        # INITIAL STATUS
        if numfreqs != 0:
            state = testenv.reset( init_freqs[freqcount] )
            freqcount += 1
            freqcount %= numfreqs
        else:
            state = testenv.reset()
        
        status = testenv._info.copy()
        
        history[f"Iteration {i + 1}"] = {}
        history[f"Iteration {i + 1}"][0] = status

        if verbose:
            print(f"Iteration {i + 1}")
            print("---------------")
            display_status("Step 0", status)

        # STEPS TO GOAL (OR MAXSTEPS)
        for s in range(testenv.MAXSTEPS):
            action = agent.compute_action(state)
            state, _, done, info = testenv.step(action)

            status = info.copy()
            history[f"Iteration {i + 1}"][s + 1] = status
            
            if verbose:
                display_status(f"Step {s + 1}", status)
            if done:
                break

    return history

def test(path, chkpt, config):
    """
        Retrieves a trained PPO agent from the indicated path. While
        the specified workload runs in the background, the specified 
        GYM environment is created and tested with the trained agent,
        according to the indicated test configuration. If stated, 
        test results and environment history log will be saved in the
        indicated path.

        Parameters
        ----------
        path : str
            Path of folder where the trained agent checkpoints are stored.
        chkpt : int
            Number of the training checkpoint to be used.
        config : dict
            General configuration of the test environment.

        Returns
        -------
        results : dict(int, int)
            Results of test, storing the frequency count of iterations.
        history : dict(str, str, dict)
            Status history of the environment during test, for each iteration 
            and step.
    """
    import ray
    from   ray.tune.registry import register_env

    env = config['env']
    work = config['work']

    ## REGISTER ENVIRONMENT
    Env = load_env_class( gym_spec(env).entry_point )
    register_env(env, lambda config: Env(**config))

    ## TRAINED AGENT RETRIEVAL
    ray.init(ignore_reinit_error=True)

    chkptpath = path + f"/checkpoint_{chkpt}/checkpoint-{chkpt}"
    agent = get_PPOagent(env, config['envconfig'], chkptpath)

    ## BACKGROUND WORKLOAD INITIALIZATION
    workers = start_work(work, config['workconfig'])

    ## TEST ENVIRONMENT WITH TRAINED AGENT
    testenv = gym_make(env, **config['envconfig'])
    history = test_env(testenv, agent, config['testconfig'])

    ## BACKGROUND WORKLOAD KILL
    end_work(workers)

    ## SAVE RESULTS AND STATUS HISTORY
    logpath = config['testconfig'].get('logpath', None)
    if logpath is not None:
        logpath += f'/checkpoint-{chkpt}'
        generate_log(history, logpath)
        save_config(config, logpath)

    return history

#########################
# --------------------- #
#########################





#########################
### COMMAND INTERFACE ###
#########################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    ## TRAINED AGENT
    agent_help = "Path of folder where the agent training checkpoints are "
    agent_help += "stored."
    parser.add_argument(
        'agent', 
        help = agent_help
    )

    ## CHECKPOINT.
    checkpoint_help = "Number of the training checkpoint to be used."
    parser.add_argument(
        'checkpoint', 
        help = checkpoint_help
    )

    ## LOAD AGENT CONFIGURATION.
    loadconfig_help = "Uses 'config.json' file generated from the agent training."
    parser.add_argument(
        '--loadconfig', 
        help = loadconfig_help,
        action = 'store_true'
    )

    ## GENERAL CONFIGURATION
    genconfig_help = "JSON file with the script general configuration."
    parser.add_argument(
        '-g', '--config', metavar='config', 
        help = genconfig_help,
        type = read_json,
        default = DEF_CONFIG
    )

    ## PARTICULAR CONFIGURATIONS
    # Environment ID
    env_help = "ID name of GYM environment the agent will use for training."
    #env_help += "CPUEnv00-v0 CPUEnv01-v0 CPUEnv02-v0."
    parser.add_argument(
        '-e' '--env', metavar = 'env', 
        help = env_help,
        type = str,
        default = argparse.SUPPRESS
    )

    # Work ID
    work_help = "ID name of operation executing in background during training."
    #work_help += "intsort intscalar floatproduct floattranspose floatsort "
    #work_help += "floatscalar"
    parser.add_argument(
        '-w', '--work', metavar = 'work',
        help = work_help,
        type = str,
        default = argparse.SUPPRESS
    )

    # Environment configuration.
    envconfig_help = "JSON file with the configuration for the GYM environment."
    parser.add_argument(
        '--envconfig', metavar = 'envconfig',
        help = envconfig_help,
        type = read_json,
        default = argparse.SUPPRESS
    )

    # Work configuration.
    workconfig_help = "JSON file with the configuration for work in the background."
    parser.add_argument(
        '--workconfig', metavar = 'workconfig',
        help = workconfig_help,
        type = read_json,
        default = argparse.SUPPRESS
    )

    # Ray Tune agent configuration.
    agentconfig_help = "JSON file with the configuration for the Ray Tune agent. "
    agentconfig_help += "If none, default Ray configuration will be used."
    parser.add_argument(
        '--agentconfig', metavar = 'agentconfig',
        help = agentconfig_help,
        type = read_json,
        default = argparse.SUPPRESS
    )

    # Training process configuration.
    testconfig_help = "JSON file with the configuration for the test process."
    parser.add_argument(
        '--testconfig', metavar = 'testconfig',
        help = testconfig_help,
        type = read_json,
        default = argparse.SUPPRESS
    )

    # CPU configuration.
    cpuconfig_help = "JSON file with the socket-cores relation of CPU."
    parser.add_argument(
        '--cpuconfig', metavar = 'cpuconfig',
        help = cpuconfig_help,
        type = read_json,
        default = argparse.SUPPRESS
    )

    ## PROCESS AFFINITY
    affinity = parser.add_mutually_exclusive_group()

    affcores_help = "The cores in which the agent will be trained."
    affinity.add_argument(
        '--affcores', metavar = 'affcores', 
        help = affcores_help,
        type = int,
        nargs = '+',
        default = argparse.SUPPRESS
    )

    affsockets_help = "The sockets in whose cores the agent will be trained."
    affinity.add_argument(
        '--affsockets', metavar = 'affsockets', 
        help = affsockets_help,
        type = int,
        nargs = '+',
        default = argparse.SUPPRESS
    )
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # SET CONFIGURATION
    config = load_config(args)

    # SET TEST PROCESS AFFINITY
    if 'affcores' in args:
        os.sched_setaffinity(0, args.affcores)
    elif 'affsockets' in args:
        cores = get_cores(args.affsockets, config['cpuconfig'])
        os.sched_setaffinity(0, cores)

    ## TEST
    test(
        path   = args.agent,
        chkpt  = args.checkpoint,
        config = config
    )


if __name__ == '__main__':
    main()
