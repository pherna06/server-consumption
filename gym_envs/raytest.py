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

# GYM CUSTOM ENVIRONMENTS
GYMENVS = {
    'CPUEnv00-v0' : CPUEnv00,
    'CPUEnv01-v0' : CPUEnv01,
    'CPUEnv02-v0' : CPUEnv02
}

DEF_CONFIG = {
    # POSITIONAL ARGUMENTS
    'env'  : '',
    'work' : '',
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
    # DEFAULT WORKLOAD CONFIGURATIONgit 
    'workconfig' : {
        'size'   : 1000,
        'groups' : [[core] for core in [8,9,10,11,12,13,14,15]]
    },
    # DEFAULT TEST CONFIGURATION
    'testconfig' : {
        'iter'      : 10,
        'logpath'   : 'tests/default',
        'verbose'   : 1
    }
}

PARTICULAR_CONFIGS = [
    'cpuconfig',
    'envconfig',
    'workconfig',
    'testconfig'
]

def load_config(args):
    # DEFAULT CONFIG LOADED FIRST
    config = DEF_CONFIG

    # DEFAULT TRAINED AGENT CONFIG
    if args.default:
        trainconfig = read_json(args.agentpath + '/config.json')
        for field in trainconfig:
            config[field] = trainconfig[field]
        
    if args.personal and 'config' in args:
        for field in args.config:
            config[field] = args.config[field]
    
    # FIELDS MODIFIED WITH THOSE OF PARTICULAR CONFIGS
    argsdict = vars(args)
    for field in PARTICULAR_CONFIGS:
        if field in args:
            config[field] = argsdict[field]

    return config

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

def generate_log(results, history, path):
    """
        Given the results and the status history log of an agent test, a CSV 
        file is generated with the results and a log file is generated with
        the status history of the environment during the test iterations. Both
        files are generated in the specified path.

        Parameters
        ----------
        results : dict(int, int)
            Results of test, storing the frequency count of iterations.
        history : dict(str, str, dict)
            Status history of the environment during test, for each iteration 
            and step.
        path : str
            Path where files will be created (overwritten if necessary)
        
    """
    respath  = path + '/results.csv'
    histpath = path + '/test.log'

    ## RESULTS CSV
    results_csv(results, respath)

    ## STATUS HISTORY LOG
    history_log(history, histpath)

def results_csv(results, csvpath):
    """
        Creates or overwrites a CSV file with the frequency count results of
        the agent test in the specified path.

        Parameters
        ----------
        results : dict(int, int)
            Results of test, storing the frequency count of iterations.
        csvpath : str
            Namepath of the CSV file.
    """
    # Create or overwrite
    if os.path.exists(csvpath):
        os.remove(csvpath)
    csvf = open(csvpath, 'w')

    # Header.
    csvf.write("Frequency, Count\n")

    # Results.
    for freq in sorted(results):
        csvf.write(f"{freq}, {results[freq]}\n")

    csvf.close()

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
    logf = open(logpath, 'w')

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

def display_results(results, total):
    """
        Displays, in standard output, the final results of the test of an
        agent, that is, the frequency count.

        Parameters
        ----------
        results : dict(int, int)
            Results of test, storing the frequency count of iterations.
        total : int
            Number of events stored.
    """
    print("Results")
    print("-------------------------")
    print("Frequency (MHz)   Count\n")
    for freq in results:
        count = results[freq]
        print(f"{freq//1000:<15}   {count/total:<5}   |")
        [printf(".", end='') for _ in range((100 * count) // total]
        print("")

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
    config = ppo.DEFAULT_CONFIG.copy()
    config['log_level']  = 'WARN'
    config['env_config'] = envconfig

    agent = ppo.PPOTrainer(config, env=env)
    agent.restore(chkptpath)

    return agent

def test_env(testenv, envconfig, agent, iterations, verbose):
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
        results : dict(int, int)
            Results of test, storing the frequency count of iterations.
        history : dict(str, str, dict)
            Status history of the environment during test, for each iteration 
            and step.
    """
    iterations = config['iter']
    verbose    = config['verbose']

    results = {}
    for freq in testenv._frequencies:
        results[freq] = 0

    history = {}

    # TEST ITERATIONS
    for i in range(iterations):
        # INITIAL STATUS
        state = testenv.reset()
        status = testenv.status()
        
        history[f"Iteration {i + 1}"] = {}
        history[f"Iteration {i + 1}"]["Step 0"] = status

        if verbose:
            print(f"Iteration {i + 1}")
            print("---------------")
            display_status("Step 0", status)

        # STEPS TO GOAL (OR MAXSTEPS)
        for s in range(testenv.MAXSTEPS):
            action = agent.compute_action(state)
            testenv.step(action)

            status = testenv.status()
            history[f"Iteration {i + 1}"][f"Step {s + 1}"] = status
            
            if verbose:
                display_status(f"Step {i + 1}")
            if status['done']:
                break

        # RECORD FREQUENCY OF FINAL STATE
        freq = status['frequency']
        results[freq] += 1

    # DISPLAY FREQUENCY COUNT RESULTS
    display_results(results, iterations)

    return results, history

def test(env, work, path, chkpt, config):
    """
        Retrieves a trained PPO agent from the indicated path. While
        the specified workload runs in the background, the specified 
        GYM environment is created and tested with the trained agent,
        according to the indicated test configuration. If stated, 
        test results and environment history log will be saved in the
        indicated path.

        Parameters
        ----------
        env : str
            Name of the GYM environment to use for the agent.
        work : str
            Name of the workload operation to be run in the background.
        path : str
            Path of folder where the trained agent checkpoints are stored.
        chkpt : int
            Number of the training checkpoint to be used.
        testconfig : dict
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

    ## REGISTER ENVIRONMENT
    Env = GYMENVS[env]
    register_env(env, lambda config: Env(**config))

    ## TRAINED AGENT RETRIEVAL
    ray.init(ignore_reinit_error=True)

    chkptpath = path + f"/checkpoint_{chkpt}/checkpoint-{chkpt}"
    agent = get_PPOagent(env, config['envconfig'], chkptpath)

    ## BACKGROUND WORKLOAD INITIALIZATION
    workers = start_work(work, config['workconfig'])

    ## TEST ENVIRONMENT WITH TRAINED AGENT
    testenv = gym.make(env, **config['envconfig'])
    results, history = test_env(testenv, agent, config['testconfig'])

    ## BACKGROUND WORKLOAD KILL
    end_work(workers)

    ## SAVE RESULTS AND STATUS HISTORY
    logpath = config['testconfig'].get('logpath', None)
    if logpath is not None:
        generate_log(results, history, logpath)

    return results, history

#########################
# --------------------- #
#########################





#########################
### COMMAND INTERFACE ###
#########################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    # Choose environment, work and agent configuration.
    inputagent = parser.add_mutually_exclusive_group(required=True)

    ## FROM TRAINED AGENT SAVED CONFIGURATION.
    defconfig_help = "Uses config.json file generated from the agent training."
    inputagent.add_argument(
        '--default', help=defconfig_help,
        action='store_true'
    )

    ## FROM SPECIFIED INPUT CONFIG FILES.
    personal_help = "Uses the configuration specified directly in the command."
    inputagent.add_argument(
        '--personal', help=personal_help,
        action='store_true'
    )

    ## GENERAL CONFIGURATION.
    genconfig_help = "Dict with the general configuration for the script."
    parser.add_argument(
        '-g', '--config', metavar='config', help=genconfig_help,
        type=read_json,
        defaul=DEF_CONFIG
    )

    ## GYM ENVIRONMENT.
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

    ## TRAINED AGENT PATH
    trainpath_help = "Path of folder where the agent training checkpoints are "
    trainpath_help += "stored."
    parser.add_argument('trainpath', help=trainpath_help)

    ## CHECKPOINT SELECTION.
    checkpoint_help = "Number of the training checkpoint to be used."
    parser.add_argument('chkpt', help=checkpoint_help)

    ## TEST CONFIGURATION
    testconfig_help = "Dict of values for the configuration of agent testing."
    parser.add_argument(
        '-x', '--testconfig', metavar='testconfig', help=testconfig_help,
        type=json.loads,
        default=DEF_TESTCONFIG
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
    env = config['env']
    if env == '':
        env = args.env
    
    work = config['work']
    if work == '':
        work = args.work

    # SET TEST PROCESS AFFINITY
    if 'affcores' in args:
        os.sched_setaffinity(0, args.affcores)
    elif 'affsockets' in args:
        os.sched_setaffinity(0, get_cores(args.affsockets))

    ## TEST
    test(
        env    = env,
        work   = work,
        path   = args.agentpath,
        chkpt  = args.chkpt,
        config = config
    )


if __name__ == '__main__':
    main()
