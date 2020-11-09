from gym.envs.registration import register

register(
    id='CPUEnv-v0',
    entry_point='gymcpu.envs:CPUEnv',
)