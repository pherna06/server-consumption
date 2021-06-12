from gym.envs.registration import register

register(
	id='CPUEnv00-v0',
	entry_point='gymcpu.envs:CPUEnv00',
)

register(
	id='CPUEnv01-v0',
	entry_point='gymcpu.envs:CPUEnv01',
)

register(
	id='CPUEnv02-v0',
	entry_point='gymcpu.envs:CPUEnv02',
)

register(
	id='CPUEnv03-v0',
	entry_point='gymcpu.envs:CPUEnv03'
)

register(
	id='FinalEnv01-v0',
	entry_point='gymcpu.envs:FinalEnv01'
)

register(
	id='FinalEnv02-v0',
	entry_point='gymcpu.envs:FinalEnv02'
)

register(
	id='FinalEnv03-v0',
	entry_point='gymcpu.envs:FinalEnv03'
)