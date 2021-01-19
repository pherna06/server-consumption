import gym
import gymcpu
import sys
import argparse

def run_test(env, verbose=False):
    env.reset()
    sum_reward = 0

    for i in range(env.MAXSTEPS):
        action = env.action_space.sample()

        if verbose:
            print('action:', action)

        state, reward, done, info = env.step(action)
        sum_reward += reward

        if verbose:
            env.render()

        if done:
            if verbose:
                print(f"done @ step {i}")
            break
    
    if verbose:
        print('cumulative reward', sum_reward)

    return sum_reward

def get_parser():
    # Parser description and creation.
    desc = "A command interface that to test (not train) GYM environments."
    parser = argparse.ArgumentParser(description = desc)

    env_help = "GYM environment name to be tested."
    parser.add_argument(
            'env', help=env_help,
            type=str
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    kwargs = {
        'socket': 0,
        'cores': [0,1,2,3,4,5,6,7],
        'limit': 40,
    }

    env = gym.make(args.env, **kwargs)
    run_test(env, verbose=True)

if __name__ == "__main__":
    main()
