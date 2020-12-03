from gymcpu.envs.cpu_env import CPUEnv
from ray.tune.registry import register_env
import gym
import gymcpu
import os
import sys
import ray
import ray.rllib.agents.ppo as ppo
import shutil
import pyRAPL

argslen = len(sys.argv)
_socket = None
_limit = None
if argslen == 3:
    _socket = int(sys.argv[1])
    _limit = int(sys.argv[2])
else:
    print("ERROR: not enough arguments.")
    exit()

def main():
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    ray.init(ignore_reinit_error=True)

    select_env = "CPUEnv-v0"
    register_env(select_env, lambda env_config: CPUEnv(**env_config))

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_workers"] = 0
    config["env_config"] = {
            'socket': _socket,
            'limit': _limit,
            'time': 0.001
            }
    agent = ppo.PPOTrainer(config, env=select_env)

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

    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    agent.restore(chkpt_file)

    cpuenv = gym.make(select_env, **config['env_config'])

    state = cpuenv.reset()
    sum_reward = 0

    n_step = 20

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = cpuenv.step(action)
        sum_reward += reward

        cpuenv.render()

        if done == 1:
            print("cumulative reward", sum_reward)
            state = cpuenv.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()
