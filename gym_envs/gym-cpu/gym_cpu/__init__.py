from gym.envs.registration import register

register(
    id='cpu_env',
    entry_point='gym_cpu.envs:CPUEnv'
)