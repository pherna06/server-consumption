import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cpufreq
import pyRAPL
import time
import numpy as np

class CPUEnv00(gym.Env):
    ### DEFAULT PERSONAL VALUES
    DEF_SOCKET = 0
    DEF_CORES = [0,1,2,3,4,5,6,7]

    DEF_MAXSTEPS = 20
    DEF_SEED = None
    DEF_TIME = 1

    DEF_POWERLIMIT = 65

    def __init__(self, 
                 socket=DEF_SOCKET, 
                 cores=DEF_CORES, 
                 limit=DEF_POWERLIMIT, 
                 steps=DEF_MAXSTEPS, 
                 time=DEF_TIME, 
                 seed=DEF_SEED):     
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
        
        ### Default metadata.
        self.metadata = { 'render.modes': ['human'] }
        
        ### Frequency control.
        #   _cpu: cpufreq class control
        #   _frequencies: list of available frequencies (<= order)
        self._cpu = cpufreq.cpuFreq()
        self._frequencies = sorted( self._cpu.available_frequencies )

        # Set scheme of used cores to 'userspace' to allow frequency control.
        self._cpu.set_governors('userspace', self.CORES) 
       
        ### Action space [0,1] where:
        #   '0' lower frequency
        #   '1' raise frequency
        self.action_space = gym.spaces.Discrete(2)
        
        self.LOWER_FREQ = 0
        self.RAISE_FREQ = 1
       
        ### Action rewards:
        #   REWARD_LOWER_BELOW lower frequency while below power limit
        #   REWARD_RAISE_BELOW raise frequency while below power limit
        #   REWARD_LOWER_ABOVE lower frequency while above power limit
        #   REWARD_RAISE_ABOVE raise frequency while above power limit
        self.REWARD_LOWER_BELOW = -1
        self.REWARD_RAISE_BELOW =  1
        self.REWARD_LOWER_ABOVE =  5
        self.REWARD_RAISE_ABOVE = -5
        
        ### Observation space:
        #   Positions of CPU frequencies list.
        self.observation_space = gym.spaces.Discrete(len(self._frequencies) + 1)

        self._state = -1

        ### CPUEnv: random number generator.
        #   _seed: numeric seed for environment rng
        #   _rng: random number generator
        self._seed = seed
        self._rng = None

        self.seed( self._seed )
                
        ### CPUEnv: general environment variables.
        #   _reward: accumulated environment reward
        #   _done: boolean value to indicate if goal or max steps were reached
        #   _info: dict for auxiliary debug values
        #   _count: counts the number of steps taken during environment action
        #   _power: current power consumption
        self._reward = None
        self._done = None
        self._info = None
        self._count = None
        self._power = None

        self.reset()

    def reset(self):        
        ### General environment variables.
        self._reward = 0
        self._done = False
        self._info = {}
        self._count = 0

        ### Set random initial frequency (state) and measure power.
        self._state = self._rng.choice( np.arange( len(self._frequencies) ) )
        self._state += 1
        freq = self._frequencies[ self._state - 1 ]
        self.set_frequency(freq)

        pyRAPL.setup( devices=[pyRAPL.Device.PKG], socket_ids=[self.SOCKET] )
        self._power = self.measure_power('Reset')

        return self._state

    def step(self, action):        
        ### Check if max steps reached.
        if self._count == self.MAXSTEPS:
            self._done = True
            return self._state, self._reward, self._done, self._info

        assert self.action_space.contains(action)
        
        # Boolean for below/above limit.
        overlimit = True if self._power > self.LIMIT else False

        ### ACTION:
        if action == self.RAISE_FREQ:
            if self._state == len(self._frequencies):
                pass
            else:
                self._state += 1
                freq = self._frequencies[ self._state - 1 ]
                self.set_frequency(freq)
        elif action == self.LOWER_FREQ:
            if self._state == 1:
                pass
            else:
                self._state -= 1
                freq = self._frequencies[ self._state - 1 ]
                self.set_frequency(freq)

        # Measure new power consumption.
        next_power = self.measure_power(f"Iter {self._count + 1}")

        ### REWARDS:
        if action == self.RAISE_FREQ:
            self._reward += self.REWARD_RAISE_ABOVE if overlimit else self.REWARD_RAISE_BELOW
        elif action == self.LOWER_FREQ:
            self._reward += self.REWARD_LOWER_ABOVE if overlimit else self.REWARD_LOWER_BELOW

        ### GOAL:
        #   if overlimit, action was lower freq and new power is below limit
        if action == self.LOWER_FREQ and overlimit and next_power < self.LIMIT:
            self._done = True

        ### INFO AND STATE UPDATE:
        self._info['delta'] = next_power - self._power
        self._info['power'] = next_power
        self._info['frequency'] = self._frequencies[ self._state - 1 ]
        self._power = next_power

        ### RETURN:
        self._count += 1
        return [self._state, self._reward, self._done, self._info]

    
    def render(self, mode='human'):
        ### Print current environment state info.
        print(
            f"state (position): {self._state},",
            f"reward: {self._reward},",
            f"info: {self._info}"
        )

    def seed(self, seed=None):
        ### Make random number generator from seed.
        self._rng, seed = seeding.np_random(seed)

        return [seed]

    def close(self):
        ### Reset CPU to default system values.
        self._cpu.reset()

    ### AUXILIARY METHODS
    
    def set_frequency(self, freq):
        ### Check if current frequency is above or below.
        if self._cpu.get_min_freq()[ self.CORES[0] ] < freq:
            # Above
            self._cpu.set_max_frequencies(freq, self.CORES)
            self._cpu.set_min_frequencies(freq, self.CORES)
        else:
            # Below
            self._cpu.set_min_frequencies(freq, self.CORES)
            self._cpu.set_max_frequencies(freq, self.CORES)

        self._cpu.set_frequencies(freq, self.CORES)


    def measure_power(self, label):
        meter = pyRAPL.Measurement(label=label)
        meter.begin()
        time.sleep(self.TIME)
        meter.end()

        m_energy = meter._results.pkg[self.SOCKET] # micro-J
        m_time = meter._results.duration # micro-s
        power = m_energy / m_time # watts

        return power
