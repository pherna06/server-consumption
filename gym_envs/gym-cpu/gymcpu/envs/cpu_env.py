import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cpufreq
import pyRAPL
import time

class CPUEnv(gym.Env):
    # POWER LIMIT
    _socket = -1
    _corelist = -1
    _limit = -1
    
    # CPU utilities
    _cpu = cpufreq.cpuFreq()
    _frequencies = sorted(_cpu.available_frequencies)
    _current_power = 0
    
    # Possible actions
    LOWER_FREQ = 0
    RAISE_FREQ = 1

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
        # Action space [0,1,2] where:
        #   '0' lower frequency
        #   '1' keep frequency
        #   '2' raise frequency
        self.action_space = gym.spaces.Discrete(2)

        # Observation space 'self.frequencies':
        self.observation_space = gym.spaces.Discrete(self.MAX_FREQ + 1)

        # Possible positions to choose on reset.
        self.init_positions = list(range(self.MIN_FREQ, self.MAX_FREQ))
        
        # Pseudorandom generator.
        self.seed()

    def step(self, action):
        if self.done:
            print("ERROR: stepped while done.")
        elif self.count == self.MAX_STEPS:
            self.done = True
        else:
            assert self.action_space.contains(action)
            self.count += 1

            # Checks whether over power limit before action.
            overlimit = True if self._current_power > self._limit else False

            ## Modify frequency through action.
            if action == self.RAISE_FREQ:
                if self.position == self.MAX_FREQ:
                    pass # No action.
                else:
                    self.position += 1 # Raise frequency.

                    # Modify frequency.
                    freq = self._frequencies[self.position - 1]
                    self._cpu.set_frequencies(freq, self._corelist)
                    self._cpu.set_max_frequencies(freq, self._corelist)
                    self._cpu.set_min_frequencies(freq, self._corelist)
            elif action == self.LOWER_FREQ:
                if self.position == self.MIN_FREQ:
                    pass # No action
                else:
                    self.position -= 1 # Lower frequency.

                    # Modify frequency.
                    freq = self._frequencies[self.position - 1]
                    self._cpu.set_frequencies(freq, self._corelist)
                    self._cpu.set_min_frequencies(freq, self._corelist)
                    self._cpu.set_max_frequencies(freq, self._corelist)

            ## Measure new power consumption.
            meter = pyRAPL.Measurement(label=f"Iter {self.count}")
            meter.begin()
            time.sleep(1) # Sleep for a second while CPU works in the background.
            meter.end()
            
            m_energy = meter._results.pkg[self._socket] # micro-J
            m_time = meter._results.duration # micro-s
            m_power = m_energy / m_time # watts

            ## Apply rewards
            if action == self.RAISE_FREQ:
                self.reward += self.REWARD_RAISE_ABOVE if overlimit else self.REWARD_RAISE_BELOW
            elif action == self.LOWER_FREQ:
                self.reward += self.REWARD_LOWER_ABOVE if overlimit else self.REWARD_LOWER_BELOW

            ## Update env values.
            self.info['delta'] = m_power - self._current_power
            self.info['power'] = m_power
            self._current_power = m_power
            self.state = self.position

            ## Check goal reached.
            if action == self.LOWER_FREQ and overlimit and self._current_power <= self._limit:
                self.done = True

            return [self.state, self.reward, self.done, self.info]

    def reset(self):
        self.position = self.np_random.choice(self.init_positions)
        self.count = 0

        # Getting CPU to initial frequency.
        freq = self._frequencies[self.position - 1]
        self._cpu.set_frequencies(freq, self._corelist)
        if self._cpu.get_min_freq()[self._corelist] < freq:
            self._cpu.set_max_frequencies(freq, self._corelist)
            self._cpu.set_min_frequencies(freq, self._corelist)
        else:
            self._cpu.set_min_frequencies(freq, self._corelist)
            self._cpu.set_max_frequencies(freq, self._corelist)

        # Measure initial power.
        meter = pyRAPL.Measurement(label=f"Reset")
        meter.begin()
        time.sleep(1) # Sleep for a second while CPU works in the background.
        meter.end()

        m_energy = meter._results.pkg[self._socket] # micro-J
        m_time = meter._results.duration # micro-s
        self._current_power = m_energy / m_time # watts

        # state is frequency position
        self.state = self.position
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def render(self, mode='human'):
        print(f"frequency: {self._frequencies[self.position - 1]:>7},",
              f"reward: {self.reward:>3},",
              f"info: {self.info}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    def set_rapl(self, socket, limit, socket_size = 8):
        # Env characteristics
        self._socket = socket
        self._limit = limit
        self._corelist = list( range(socket_size * socket, socket_size * (socket + 1)) )

        # Setup pyRAPL.
        pyRAPL.setup(devices=[pyRAPL.Device.PKG], socket_ids=[self._socket])

        self.reset()