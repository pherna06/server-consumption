import gym
import gymcpu
import sys

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

def main():
    kwargs = {
        'socket': 0,
        'cores': [0,1,2,3,4,5,6,7],
        'limit': 40,
    }

    env = gym.make('CPUEnv01-v0', **kwargs)
    run_test(env, verbose=True)

if __name__ == "__main__":
    main()
