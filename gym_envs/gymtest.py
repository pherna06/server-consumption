import gymcpu
from   gym    import make

import argparse

DEF_ACTIONS = 10

def run_test(env, actions):
	"""
		Test the given GYM environment for a number of actions.

		Parameters
		----------
		env : GYM Environment
			The environment to be tested.
	"""
	env.reset()
	sum_reward = 0

	for i in range(actions):
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

	act_help = "Number of actions to perform in test. "
	act_help = "Set to {} by default.".format(DEF_ACTIONS)
	parser.add_argument(
		'-a', '--actions', help = act_help,
		type = int,
		default = DEF_ACTIONS
	)

	return parser

def main():
	parser = get_parser()
	args = parser.parse_args()

	kwargs = {
		'socket': 0,
		'cores': [0],
	}

	env = make(args.env, **kwargs)
	run_test(env, args.actions)

if __name__ == "__main__":
	main()