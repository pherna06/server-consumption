import gym
import gymcpu
import sys

def run_test(env, verbose=False):
    env.reset()
    sum_reward = 0

    for i in range(env.MAX_STEPS):
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
    env = gym.make('CPUEnv-v0')
    env.set_rapl(core, limit)
    sum_reward = run_test(env, verbose=True)

    history = []

    for _ in range(10):
        sum_reward = run_test(env)
        history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print(f"\nbaseline cumulative reward: {avg_sum_reward:6.2}")

# Handle script argv
args = len(sys.argv)
core = None
limit = None
if args == 3:
    core = int(sys.argv[1])
    limit = int(sys.argv[2])
else:
    print("ERROR: not enough arguments.")
    exit()

if __name__ == "__main__":
    main()