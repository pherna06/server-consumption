import gym
import gymcpu
from gymcpu.envs.cpu_env00 import CPUEnv00
from gymcpu.envs.cpu_env01 import CPUEnv01
from gymcpu.envs.cpu_env02 import CPUEnv02

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

import numpy as np
import subprocess
import os
import shutil
import signal
import argparse

MEASURE_TIME = 0.001 # seconds

SOCKET = 1
CORES = [8,9,10,11,12,13,14,15]

MATRIX_SIZE = 1000

MAX_INT = 10000000

########################
### NUMPY OPERATIONS ###
########################

def int_product(size):
    """
        Performs the product of random integer matrices in an infinite loop.

        Parameters
        ----------
        size : int
            The dimension of the square matrices that will be used in the
            operation.
    """
    # Random matrix generation.
    matA = np.random.randint(MAX_INT, size=(size, size))
    matB = np.random.randint(MAX_INT, size=(size, size))

    while(True):
        matC = np.matmul(matA, matB)


def float_product(size):
    """
        Performs the product of random real matrices in an infinite loop.

        Parameters
        ----------
        size : int
            The dimension of the square matrices that will be used in the
            operation.
    """
    # Random matrix generation.
    matA = np.random.rand(size, size)
    matB = np.random.rand(size, size)    

    while(True):
        matC = np.matmul(matA, matB)


def int_transpose(size):
    """
        Performs the transposition of a random integer matrix in an infinite
        loop. The numpy copy() method is used so that transposition is done 
        'physically' in memory; otherwise numpy would permorm it in constant
        time by swapping axes.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
    """
    # Random matrix generation.
    matA = np.random.randint(MAX_INT, size=(size, size))

    while(True):
        matA = matA.transpose().copy()


def float_transpose(size):
    """
        Performs the transposition of a random real matrix in an infinite
        loop. The numpy copy() method is used so that transposition is done 
        'physically' in memory; otherwise numpy would permorm it in constant
        time by swapping axes.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
    """
    # Random matrix generation.
    matA = np.random.rand(size, size)

    while(True):
        matA = matA.transpose().copy()


def int_sort(size):
    """
        Performs the sorting of a random integer array in an infinite loop.

        Parameters
        ----------
        size : int
            The squared-root value of the size of the array that will be used
            in the operation. That is, the number of elements in the array is
            size * size.
    """
    # Random array generation
    arrayA = np.random.randint(MAX_INT, size=(size*size))

    while(True):
        arrayB = np.sort(arrayA)


def float_sort(size):
    """
        Performs the sorting of a random real array in an infinite loop.

        Parameters
        ----------
        size : int
            The squared-root value of the size of the array that will be used
            in the operation. That is, the number of elements in the array is
            size * size.
    """
    # Random array generation
    arrayA = np.random.rand(size*size)

    while(True):
        arrayB = np.sort(arrayA)


def int_scalar(size):
    """
        Performs the sum of a random integer to each element of a random 
        integer matrix in an infinite loop.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
    """
    # Random matrix generation
    matA = np.random.randint(MAX_INT, size=(size, size))
    intN = np.random.randint(MAX_INT)

    while(True):
        matB = matA + intN

def float_scalar(size):
    """
        Performs the sum of a random real number to each element of a random 
        real matrix in an infinite loop.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
    """
    # Random matrix generation
    matA = np.random.rand(size, size)
    floatN = np.random.rand()

    while(True):
        matB = matA + floatN



OPERATIONS = {
    'intproduct':       int_product,
    'inttranspose':     int_transpose,
    'intsort':          int_sort,
    'intscalar':        int_scalar,
    'floatproduct':     float_product,
    'floattranspose':   float_transpose,
    'floatsort':        float_sort,
    'floatscalar':      float_scalar
}


#############
### FORKS ###
#############

def power_fork(work, core, size):
    """
        power_fork is used to handle forked processes for energy measure.
        The affinity of the process is set to :core:. The operation to be
        executed and measured is given by :work:.
        The forked process will not return; it has to be killed by the
        parent process.
    """
    # Set core affinity.
    pid = os.getpid()
    command = "taskset -cp " + str(core) + " " + str(pid)
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)

    # Start work operation:
    op = OPERATIONS[work]
    op(size) ## -> INFINITE LOOP


####################
### ENVIRONMENTS ###
####################

ENVIRONMENTS = {
    'CPUEnv00-v0':      CPUEnv00,
    'CPUEnv01-v0':      CPUEnv01,
    'CPUEnv02-v0':      CPUEnv02
}


def train_env(env, power, time):
    # Init training environment.
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    ray.init(ignore_reinit_error=True)

    # Config agent.
    Env = ENVIRONMENTS[env]
    register_env(env, lambda env_config: Env(**env_config))

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_workers"] = 0
    config["env_config"] = {
        'socket': SOCKET,
        'cores': CORES,
        'limit': power,
        'time': time
    }
    agent = ppo.PPOTrainer(config, env=env)

    # Train agent.
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 5

    for n in range (n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
            ))

    # Print policy.
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    # Restore and test environment.
    agent.restore(chkpt_file)
    cpuenv = gym.make(env, **config['env_config'])

    state = cpuenv.reset()
    sum_reward = 0
    n_step = 20
    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, _ = cpuenv.step(action)
        sum_reward += reward

        cpuenv.render()

        if done:
            print("cumulative reward", sum_reward)
            state = cpuenv.reset()
            sum_reward = 0

def get_parser():
    # Parser description and creation.
    desc = "A command interface that implements OpenAI Gym environments "
    desc += "training with RLlib."
    parser = argparse.ArgumentParser(description = desc)

    # Parser environment variable.
    env_help = "The name of a GYM environment defined within gymcpu local module. "
    env_help += "Available environments:"
    env_help += "".join(f" {env}" for env in ENVIRONMENTS)
    parser.add_argument(
        'env', help=env_help,
        metavar='ENV',
        type=str
    )

    # Parser workload variable.
    work_help = "The name of the operation to be performed by the CPU while "
    work_help += "training the environment."
    work_help += "Available works:"
    work_help += "".join( f" {work}" for work in OPERATIONS)
    parser.add_argument(
        'work', help=work_help,
        metavar='WORK',
        type=str
    )

    # Parser power limit variable.
    power_help = "The power limit for the environment, float, in watts."
    parser.add_argument(
        'power', help=power_help,
        metavar='POW',
        type=float,
    )

    # Parser argument for measure time.
    time_help = "Time used by pyRAPL to measure CPU power in each step "
    time_help += "of the environment action. "
    time_help += f"Set to {MEASURE_TIME} by default."
    parser.add_argument(
        '-t', '--time', help=time_help,
        type=float,
        default=MEASURE_TIME
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    env = args.env
    work = args.work
    power = args.power
    time = args.time

    # Fork workload to socket 1.
    pidls = []
    for core in CORES:
        pid = os.fork()
        if pid == 0:
            power_fork(work, core, MATRIX_SIZE)
        else:
            pidls.append(pid)

    # Set measuring process in socket 0
    mainpid = os.getpid()
    command = "taskset -cp 0-7 " + str(mainpid)
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)

    train_env(env, power, time)

    # Kill forks.
    for pid in pidls:
        os.kill(pid, signal.SIGKILL)
    
if __name__ == '__main__':
    main()


