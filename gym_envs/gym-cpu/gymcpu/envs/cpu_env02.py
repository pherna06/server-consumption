import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cpufreq
import pyRAPL
import time
import numpy as np
from math import ceil

class CPUEnv02(gym.Env):
    ### DEFAULT PERSONAL VALUES
    DEF_SOCKET = 0
    DEF_CORES = [0,1,2,3,4,5,6,7]

    DEF_MAXSTEPS = 20
    DEF_SEED = None
    DEF_TIME = 1
    
    DEF_MINPOWER = 15.0 - 5.0
    DEF_MAXPOWER = 115.0 + 5.0

    DEF_POWERSTEP = 2.0 

    DEF_POWERLIMIT = 65.0

    def __init__(self,
                 socket=DEF_SOCKET,
                 cores=DEF_CORES,
                 limit=DEF_POWERLIMIT,
                 steps=DEF_MAXSTEPS,
                 time=DEF_TIME,
                 seed=DEF_SEED,
                 minpow=DEF_MINPOWER,
                 maxpow=DEF_MAXPOWER,
                 powstep=DEF_POWERSTEP
                 ):
        ### CPUEnv constant values.
        #   SOCKET socket to get pyRAPL measures
        #   CORES CPU cores assigned to SOCKET
        #   LIMIT power limit for environment functionality
        #   MAXSTEPS maximum iterations for environment
        #   TIME time spent in each rapl measurement
        #   POWERSTEP size of intervals of observation space
        self.SOCKET = socket
        self.CORES = cores
        self.LIMIT = limit
        self.MAXSTEPS = steps
        self.TIME = time
        self.MINPOWER = minpow
        self.MAXPOWER = maxpow
        self.BANDWIDTH = self.MAXPOWER - self.MINPOWER
        self.POWERSTEP = powstep
        self.INTERVALS =  ceil( (self.BANDWIDTH) / self.POWERSTEP )

        ### Default metadata.
        self.metadata = { 'render.modes': ['human'] }

        ### Frequency control.
        #   _cpu: cpufreq class control
        #   _frequencies: list of available frequencies (<= order)
        #   _freqpos: position of current frequency
        self._cpu = cpufreq.cpuFreq()
        self._frequencies = sorted( self._cpu.available_frequencies )
        self._freqpos = -1

        # Set used cores to 'userspace' scheme for frequency modification.
        self._cpu.set_governors('userspace', cores)

        ### Action space.
        #   0: lower frequency
        #   1: raise frequency
        self.action_space = gym.spaces.Discrete(2)

        self.LOWER_FREQ = 0
        self.RAISE_FREQ = 1

        ### Action rewards:
        #   REWARD_LOWER_BELOW lower frequency while below limit interval
        #   REWARD_RAISE_BELOW raise frequency while below limit interval
        #   REWARD_LOWER_ABOVE lower frequency while above limit interval
        #   REWARD_RAISE_ABOVE raise frequency while above limit interval
        #   REWARD_GOAL interval goal reached
        self.REWARD_LOWER_BELOW = -1
        self.REWARD_RAISE_BELOW =  1
        self.REWARD_LOWER_ABOVE =  5
        self.REWARD_RAISE_ABOVE = -5
        self.REWARD_GOAL = 50
       
        ### Observation space:
        #   Interval partition of power range of CPU.
        #   Shape of intervals: (power_i, power_i+1]
        self.observation_space = gym.spaces.Discrete(self.INTERVALS)
        
        #   _power: current power consumption
        #   _state: interval of current power consumption
        #   _goal: interval of self.LIMIT
        self._power = 0.0
        self._state = 0
        self._goal = ceil((self.LIMIT - self.MINPOWER) / self.POWERSTEP)
        
        ### CPUEnv: random number generator.
        #   _seed: numeric seed for the enviroment rng
        #   _rng: random number generator
        self._seed = seed
        self._rng = None

        self.seed( self._seed )

        ### CPUEnv: general environment variables.
        #   _reward: accumulated environment reward
        #   _done: boolean value to indicate if goal or max steps were reached
        #   _info: dict for auxiliary debug values
        #   _count: counts the number of steps taken during environment action
        self._reward = None
        self._done = None
        self._info = None
        self._count = None

        self.reset()

    def reset(self):
        ### General environment variables.
        self._reward = 0
        self._done = False
        self._info = {}
        self._count = 0

        ### Set random initial frequency.
        self._freqpos = self._rng.choice( np.arange( len(self._frequencies) ) )
        freq = self._frequencies[ self._freqpos ]
        self.set_frequency(freq)

        ### Measure power and calculate power interval (state)
        pyRAPL.setup( devices=[pyRAPL.Device.PKG], socket_ids=[self.SOCKET] )
        self._power = self.measure_power('Reset')

        self._state = self.get_state( self._power )

        return self._state        

    def step(self, action):        
        ### Check if max steps reached.
        if self._count == self.MAXSTEPS:
            self._done = True
            return self._state, self._reward, self._done, self._info

        assert self.action_space.contains(action)

        ### ACTION:
        if action == self.RAISE_FREQ:
            if self._freqpos == len(self._frequencies) - 1:
                pass
            else:
                self._freqpos += 1
                freq = self._frequencies[ self._freqpos ]
                self.set_frequency(freq)
        elif action == self.LOWER_FREQ:
            if self._freqpos == 0:
                pass
            else:
                self._freqpos -= 1
                freq = self._frequencies[ self._freqpos ]
                self.set_frequency(freq)

        # Measure new power consumption.
        next_power = self.measure_power(f"Iter {self._count + 1}")
        next_state = self.get_state( next_power )

        ### REWARDS:
        diff = next_state - self._goal
        if diff == 0:
            self._reward += self.REWARD_GOAL
        else:
            if action == self.RAISE_FREQ:
                if diff > 0:
                    self._reward += self.REWARD_RAISE_ABOVE
                else:
                    self._reward += self.REWARD_RAISE_BELOW
            elif action == self.LOWER_FREQ:
                if diff > 0:
                    self._reward += self.REWARD_LOWER_ABOVE
                else:
                    self._reward += self.REWARD_LOWER_BELOW

        ### GOAL:
        #   if measured power interval is the same as LIMIT interval (_goal)
        if next_state == self._goal:
            self._done = True

        ### INFO AND STATE UPDATE:
        self._info['delta'] = next_power - self._power
        self._info['power'] = next_power
        self._info['frequency'] = self._frequencies[ self._freqpos ]
        lowpow = (next_state - 1) * self.POWERSTEP
        self._info['lower_bound'] = self.MINPOWER + lowpow
        self._info['higher_bound'] = self._info['lower_bound'] + self.POWERSTEP
        self._power = next_power
        self._state = next_state

        ### RETURN:
        self._count += 1
        return [self._state, self._reward, self._done, self._info]

    def render(self, mode='human'):
        ### Print current environtment state info.
        print(
            f"interval: {self._state}, ",
            f"reward: {self._reward}, ",
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

    def get_state(self, power):
        abspow = power - self.MINPOWER
        return ceil(abspow / self.POWERSTEP)

    def set_frequency(self, freq):
        ### Check if current frequency is above or below
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