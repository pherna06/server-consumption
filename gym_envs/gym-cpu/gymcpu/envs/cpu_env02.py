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
    DEF_POWERLIMIT = 65.0

    DEF_SOCKET = 0
    DEF_CORES  = [0,1,2,3,4,5,6,7]

    DEF_MAXSTEPS = 20
    DEF_SEED     = None
    
    DEF_MINPOWER = 15.0
    DEF_MAXPOWER = 115.0

    DEF_POWERSTEP = 2.0 

    DEF_DECISION = 0.25 # 4 decisions / sec

    def __init__(self, **config):
        ### CPUEnv constant values.
        #   SOCKET socket to get pyRAPL measures
        #   CORES CPU cores assigned to SOCKET
        #   LIMIT power limit for environment functionality
        #   MAXSTEPS maximum iterations for environment
        #   TIME time spent in each rapl measurement
        #   POWERSTEP size of intervals of observation space
        self.LIMIT  = config.get('power',  self.DEF_POWERLIMIT)

        self.SOCKET = config.get('socket', self.DEF_SOCKET)
        self.CORES  = config.get('cores',  self.DEF_CORES)

        self.MAXSTEPS = config.get('maxsteps', self.DEF_MAXSTEPS)
        self.SEED     = config.get('seed',     self.DEF_SEED)

        self.MINPOWER  = config.get('minpower', self.DEF_MINPOWER)
        self.MAXPOWER  = config.get('maxpower', self.DEF_MAXPOWER)
        assert(self.MINPOWER < self.MAXPOWER)

        self.DECISION_TIME = config.get('decision_time', self.DEF_DECISION)
        self.MEASURE_TIME  = config.get('measure_time',  self.DECISION_TIME)
        self.SLEEP_TIME    = self.DECISION_TIME - self.MEASURE_TIME
        assert(self.SLEEP_TIME >= 0)

        self.POWERPOINTS = self.get_powerpoints(**config)
        self.INTERVALS = len(self.POWERPOINTS) + 1

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
        self._cpu.set_governors('userspace', self.CORES)

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
        self.observation_space = gym.spaces.Discrete(self.INTERVALS + 1)
        
        #   _power: current power consumption
        #   _state: interval of current power consumption
        #   _goal: interval of self.LIMIT
        self._power = 0.0
        self._state = 0
        self._goal  = self.get_state(self.LIMIT)
        
        ### CPUEnv: random number generator.
        #   RNG random number generator
        self.RNG = None
        self.seed( self.SEED )

        ### CPUEnv: general environment variables.
        #   _reward: accumulated environment reward
        #   _done: boolean value to indicate if goal or max steps were reached
        #   _info: dict for auxiliary debug values
        #   _count: counts the number of steps taken during environment action
        self._reward = None
        self._done   = None
        self._info   = None
        self._count  = None

        self.reset()

    def reset(self):
        ### General environment variables.
        self._reward = 0
        self._done   = False
        self._info   = {}
        self._count  = 0

        ### Set random initial frequency.
        self._freqpos = self.RNG.choice( np.arange( len(self._frequencies) ) )
        freq = self._frequencies[ self._freqpos ]

        ### Measure power and calculate power interval (state)
        pyRAPL.setup( 
            devices=[pyRAPL.Device.PKG], 
            socket_ids=[self.SOCKET] 
            )

        self._power = self.set_wait_measure(freq, 'Reset')
        self._state = self.get_state( self._power )

        return self._state        

    def step(self, action):        
        ### Check if max steps reached.
        if self._count == self.MAXSTEPS:
            self._done = True
            return self._state, self._reward, self._done, self._info

        assert self.action_space.contains(action)

        ### DECIDE ACTION:
        if action == self.RAISE_FREQ:
            if self._freqpos == len(self._frequencies) - 1:
                pass
            else:
                self._freqpos += 1
        elif action == self.LOWER_FREQ:
            if self._freqpos == 0:
                pass
            else:
                self._freqpos -= 1

        ### DO ACTION, WAIT AND MEASURE:
        freq = self._frequencies[ self._freqpos ]
        label = f"Iter {self._count + 1}"
        next_power = self.set_wait_measure(freq, label)
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
        self._info['delta']     = next_power - self._power
        self._info['power']     = next_power
        self._info['frequency'] = self._frequencies[ self._freqpos ]

        #lowpow = (next_state - 1) * self.POWERSTEP
        #self._info['lower_bound']  = self.MINPOWER + lowpow
        #self._info['higher_bound'] = self._info['lower_bound'] + self.POWERSTEP

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

    def status(self):
        status = self._info

        return status

    def seed(self, seed=None):
        ### Make random number generator from seed.
        self.RNG, seed = seeding.np_random(seed)

        return [seed]

    def close(self):
        ### Reset CPU to default system values.
        self._cpu.reset()

    ### AUXILIARY METHODS

    def get_powerpoints(self, **config):
        powers = []
        if 'powpoints' in config:
            powers = sorted(config['powpoints'])
        elif 'pownum' in config:
            num = config['pownum']
            pstep = (self.MAXPOWER - self.MINPOWER) / (num + 1)
            ppoint = self.MINPOWER
            for _ in range(num + 2):
                powers.append(ppoint)
                ppoint += pstep
        else:
            pstep = config.get('powstep', self.DEF_POWERSTEP)
            ppoint = self.MINPOWER
            powers.append(ppoint)
            while ppoint < self.MAXPOWER:
                ppoint += pstep
                powers.append(ppoint)

        return powers

    def get_state(self, power):
        pos = np.searchsorted(self.POWERPOINTS, power, side='right')

        return pos + 1

    def set_frequency(self, freq):
        ### Check if current frequency is above or below
        current_freq = self._cpu.get_min_freq()[ self.CORES[0] ]
        if current_freq < freq:
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
        time.sleep(self.MEASURE_TIME)
        meter.end()

        m_energy = meter._results.pkg[self.SOCKET] # micro-J
        m_time   = meter._results.duration         # micro-s

        power = m_energy / m_time # watts

        return power

    def set_wait_measure(self, freq, label):
        self.set_frequency(freq)

        time.sleep(self.SLEEP_TIME)

        power = self.measure_power(label)

        return power
