from gym.envs.registration import register

register(
    id='CPUEnv-v0',
    entry_point='gymcpu.envs:CPUEnv',
)

register(
    id='CPUEnv01-v0',
    entry_point='gymcpu.envs:CPUEnv01',
)
