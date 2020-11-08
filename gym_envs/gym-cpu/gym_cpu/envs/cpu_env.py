import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cpufreq
import pyRAPL
import time

class CPUEnv(gym.Env):
    # POWER LIMIT
    CPU = 0
    LIMIT = 0
    
    # CPU utilities
    _cpu = cpufreq.cpuFreq()
    _frequencies = sorted(_cpu.available_frequencies)
    _current_power = 0
    
    # Possible actions
    LOWER_FREQ = 0
    KEEP_FREQ  = 1
    RAISE_FREQ = 2

    # Possible positions
    MIN_FREQ = 1
    MAX_FREQ = len(_frequencies)

    # Steps limit
    MAX_STEPS = 20

    # Rewards
    REWARD_LOWER_BELOW = -2 # Lowering frequency while below power limit penalized.
    REWARD_RAISE_BELOW = -1 # Raising frequency while below power limit slightly penalized.

    REWARD_LOWER_ABOVE = MAX_STEPS   # Lowering frequency while above power limit rewarded.
    REWARD_RAISE_ABOVE = - MAX_STEPS # Raising frequency while above power limit greatly penalized.
    
    metadata = { 'render.modes': ['human'] }

    def __init__(self):
        # Setup pyRAPL.
        pyRAPL.setup(devices=[pyRAPL.Device.PKG], socket_ids=[self.CPU])

        # Action space [0,1,2] where:
        #   '0' lower frequency
        #   '1' keep frequency
        #   '2' raise frequency
        self.action_space = gym.spaces.Discrete(3)

        # Observation space 'self.frequencies':
        self.observation_space = gym.spaces.Discrete(self.MAX_FREQ + 1)

        # Possible positions to choose on reset.
        self.init_positions = list(range(self.MIN_FREQ, self.MAX_FREQ))
        
        # Pseudorandom generator.
        self.seed()

        self.reset()

    def step(self, action):
        if self.done:
            print("ERROR: stepped while done.")
        elif self.count == self.MAX_STEPS:
            self.done = True
        else:
            assert self.action_space.contains(action)
            self.count += 1

            # Checks whether over power limit before action.
            overlimit = True if self._current_power > self.LIMIT else False

            ## Modify frequency through action.
            if action == self.RAISE_FREQ:
                if self.position == self.MAX_FREQ:
                    pass # No action.
                else:
                    self.position += 1 # Raise frequency.

                    # Modify frequency.
                    freq = self._frequencies[self.position - 1]
                    self._cpu.set_frequencies(freq, self.CPU)
                    self._cpu.set_max_frequencies(freq, self.CPU)
                    self._cpu.set_min_frequencies(freq, self.CPU)
            elif action == self.LOWER_FREQ:
                if self.position == self.MIN_FREQ:
                    pass # No action
                else:
                    self.position -= 1 # Lower frequency.

                    # Modify frequency.
                    freq = self._frequencies[self.position - 1]
                    self._cpu.set_frequencies(freq, self.CPU)
                    self.set_min_frequencies(freq, self.CPU)
                    self.set_max_frequencies(freq, self.CPU)

            ## Measure new power consumption.
            meter = pyRAPL.Measurement(label=f"Iter {self.count}")
            meter.begin()
            time.sleep(1) # Sleep for a second while CPU works in the background.
            meter.end()
            
            m_energy = meter._results.pkg # micro-J
            m_time = meter._results.duration # micro-s
            m_power = m_energy / m_time # watts

            ## Apply rewards
            if action == self.RAISE_FREQ:
                self.reward += self.REWARD_RAISE_ABOVE if overlimit else self.REWARD_RAISE_BELOW
            elif action == self.LOWER_FREQ:
                self.reward += self.REWARD_LOWER_ABOVE if overlimit else self.REWARD_LOWER_BELOW

            ## Check goal reached.
            if action == self.RAISE_FREQ and self._current_power > LIMIT and m_power <= LIMIT:
                self.done = True

            ## Update env values and return.
            self.info['delta'] = m_power - self._current_power
            self.info['power'] = m_power
            self.current_power = m_power
            self.state = self.position

            return [self.state, self.reward, self.done, self.info]

    def reset(self):
        self.position = self.np_random.choice(self.init_positions)
        self.count = 0

        # state is frequency position
        self.state = self.position
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def render(self, mode='human'):
        print(f"frequency: {self._frequencies[self.position]:>7},",
              f"reward: {self.reward:>3},",
              f"info: {self.info}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass