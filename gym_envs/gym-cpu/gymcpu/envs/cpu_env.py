import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cpufreq
import pyRAPL
import time

class CPUEnv(gym.Env):
    ### DEFAULT VALUES
    DEF_SOCKET = 0
    DEF_CORES = [0,1,2,3,4,5,6,7]
    DEF_POWERLIMIT = 100
    DEF_MAXSTEPS = 20
    DEF_SEED = None
    DEF_TIME = 1

    def __init__(self, socket=DEF_SOCKET, cores=DEF_CORES, limit=DEF_POWERLIMIT, steps=DEF_MAXSTEPS, time=DEF_TIME, seed=DEF_SEED):     
        ### Default metadata.
        self.metadata = { 'render.modes': ['humen'] }
        
        ### CPU utilities:
        self._cpu = cpufreq.cpuFreq()
        self._frequencies = sorted( self._cpu.available_frequencies )

        ### Action space [0,1] where:
        #   '0' lower frequency
        #   '1' raise frequency
        self.action_space = gym.spaces.Discrete(2)
        
        self.LOWER_FREQ = 0
        self.RAISE_FREQ = 1

        ### Observation space:
        #   Positions of CPU frequencies list.
        self.observation_space = gym.spaces.Discrete( len(self._frequencies) + 1 )

        self.MIN_FREQ = 1
        self.MAX_FREQ = len(self._frequencies)

        self.POSITIONS = list(range(self.MIN_FREQ, self.MAX_FREQ))

        ### Action rewards:
        #   REWARD_LOWER_BELOW lower frequency while below power limit
        #   REWARD_RAISE_BELOW raise frequency while below power limit
        #   REWARD_LOWER_ABOVE lower frequency while above power limit
        #   REWARD_RAISE_ABOVE raise frequency while above power limit
        self.REWARD_LOWER_BELOW = -1
        self.REWARD_RAISE_BELOW =  1
        self.REWARD_LOWER_ABOVE =  5
        self.REWARD_RAISE_ABOVE = -5

        ### CPUEnv constant values.
        #   SOCKET socket to get pyRAPL measures
        #   CORES CPU cores assigned to SOCKET
        #   LIMIT power limit for environment functionality
        #   MAXSTEPS maximum iterations for environment
        #   TIME time spent in each rapl measurement
        self.SOCKET = socket
        self.CORES = cores
        self.LIMIT = limit
        self.MAXSTEPS = steps
        self.TIME = time

        ### CPUEnv variables.
        #   _current_power last power consumption measured
        #   _seed numeric seed for the enviroment rng
        #   _position position of current frequency being used
        #   _reward accumulated reward
        #   _done boolean value to indicate if goal or max steps were reached
        #   _info dict for auxiliary debug values
        #   _count counts the number of steps taken
        self._current_power = 0.0
        self._seed = seed
        self._position = -1
        self._reward = -1
        self._done = False
        self._info = {}
        self._count = -1

        self.seed(self._seed)

    def step(self, action):        
        ### Check if max steps reached.
        if self._count == self.MAXSTEPS:
            self._done = True
            return self._position, self._reward, self._done, {}

        ### Environment iteration.
        assert self.action_space.contains(action)
        
        # Boolean for below/above limit.
        overlimit = True if self._current_power > self.LIMIT else False

        ### ACTION:
        if action == self.RAISE_FREQ:
            if self._position == self.MAX_FREQ:
                pass
            else:
                self._position += 1
                self.set_frequency(self._position)
        elif action == self.LOWER_FREQ:
            if self._position == self.MIN_FREQ:
                pass
            else:
                self._position -= 1
                self.set_frequency(self._position)

        # Measure new power consumption.
        m_power = self.measure_power(f"Iter {self._count + 1}")

        ### REWARDS:
        if action == self.RAISE_FREQ:
            self._reward += self.REWARD_RAISE_ABOVE if overlimit else self.REWARD_RAISE_BELOW
        elif action == self.LOWER_FREQ:
            self._reward += self.REWARD_LOWER_ABOVE if overlimit else self.REWARD_LOWER_BELOW

        ### GOAL:
        #   if overlimit, action was lower freq and new power is below limit
        if action == self.LOWER_FREQ and overlimit and m_power < self.LIMIT:
            self._done = True

        ### INFO:
        self._info['delta'] = m_power - self._current_power
        self._info['power'] = m_power
        self._current_power = m_power

        ### RETURN:
        self._count += 1
        return [self._position, self._reward, self._done, self._info]

    def reset(self):        
        ### Starting frequency is set.
        self._position = self._rng.choice(self.POSITIONS)
        self.set_frequency(self._position)

        ### pyRAPL setup and power measured.
        pyRAPL.setup( devices=[pyRAPL.Device.PKG], socket_ids=[self.SOCKET] )
        self._current_power = self.measure_power('Reset')

        # state is frequency position
        self._count = 0
        self._reward = 0
        self._done = False

        return self._position

    def render(self, mode='human'):
        print(
            f"frequency: {self._frequencies[self._position - 1]:>7},",
            f"reward: {self._reward:>3},",
            f"info: {self._info}"
        )

    def seed(self, seed=None):
        self._rng, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    ### AUXILIARY METHODS
    
    def set_rapl(self, socket, cores, limit):
        self.SOCKET = socket
        self.CORES = cores
        self.LIMIT = limit

    def set_frequency(self, position):
        freq = self._frequencies[position - 1]
        self._cpu.set_frequencies(freq, self.CORES)

        # Check if current frequency is above or below.
        if self._cpu.get_min_freq()[ self.CORES[0] ] < freq:
            # Above
            self._cpu.set_max_frequencies(freq, self.CORES)
            self._cpu.set_min_frequencies(freq, self.CORES)
        else:
            # Below
            self._cpu.set_min_frequencies(freq, self.CORES)
            self._cpu.set_max_frequencies(freq, self.CORES)

    def measure_power(self, label):
        meter = pyRAPL.Measurement(label=label)
        meter.begin()
        time.sleep(self.TIME)
        meter.end()

        m_energy = meter._results.pkg[self.SOCKET] # micro-J
        m_time = meter._results.duration # micro-s
        power = m_energy / m_time # watts

        return power




