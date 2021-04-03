import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cpufreq
import pyRAPL
import time
import numpy as np
from math import ceil

class CPUEnv03(gym.Env):
    ### DEFAULT PERSONAL VALUES
    DEF_POWER = 65.0

    DEF_SOCKET = 0
    DEF_CORES  = [0,1,2,3,4,5,6,7]

    DEF_MAXSTEPS = 20
    DEF_SEED     = None
    
    DEF_MINPOWER = 15.0
    DEF_MAXPOWER = 115.0

    DEF_POWERSTEP = 3.0 

    DEF_DECISION = 0.25 # 4 decisions / sec

    def __init__(self, **config):
        ### CPUEnv constant values.
        #   SOCKET      socket to get pyRAPL measures
        #   CORES       CPU cores assigned to SOCKET
        #   LIMIT       power limit for environment functionality
        #   MAXSTEPS    maximum iterations for environment
        #   TIME        time spent in each rapl measurement
        #   POWERSTEP   size of intervals of observation space
        self.POWER  = config.get('power',  self.DEF_POWER)

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
        self.INTERVALS   = self.get_intervals(self.POWERPOINTS)

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

        ### Power measure.
        pyRAPL.setup( 
            devices    = [pyRAPL.Device.PKG], 
            socket_ids = [self.SOCKET] 
        )

        ### Action space.
        #   0: hold  frequency
        #   1: lower frequency
        #   2: raise frequency
        self.action_space = gym.spaces.Discrete(3)

        self.HOLD_FREQ  = 0
        self.LOWER_FREQ = 1
        self.RAISE_FREQ = 2

        ### Action rewards:
        #   See 'get_reward()'
       
        ### Observation space:
        #   Interval partition of power range of CPU.
        #   Shape of intervals: (power_i, power_i+1]
        self.observation_space = gym.spaces.Discrete( len(self.INTERVALS) + 1 )
        
        #   _power: current power consumption
        #   _state: interval of current power consumption
        #   _goal: interval of self.LIMIT
        self._power = 0.0
        self._state = 0
        self._goal  = self.get_state(self.POWER)
        
        ### CPUEnv: random number generator.
        #   RNG random number generator
        self.RNG = None
        self.seed( self.SEED )

        ### CPUEnv: general environment variables.
        #   _reward: accumulated environment reward
        #   _done: boolean value to indicate if goal or max steps were reached
        #   _info: dict for auxiliary debug values
        #   _count: counts the number of steps taken during environment action
        self._reward     = None
        self._acc_reward = None

        self._done   = None
        self._info   = None
        self._count  = None

        self.reset()

    def reset(self):
        ### General environment variables.
        self._reward     = 0
        self._acc_reward = 0
        self._done   = False
        self._info   = {}
        self._count  = 0

        ### Choose random initial frequency.
        self._freqpos = self.RNG.choice( np.arange( len(self._frequencies) ) )
        freq = self._frequencies[ self._freqpos ]

        ### Set frequency, wait sleep time and measure.
        self._power = self.set_wait_measure(freq, 'Reset')

        ### Set state from measured power.
        self._state = self.get_state( self._power )

        return self._state

    def step(self, action):        
        ### Check if max steps reached.
        if self._count == self.MAXSTEPS:
            self._done = True
            return self._state, self._reward, self._done, self._info

        assert self.action_space.contains(action)

        ### DECIDE ACTION:
        if action == self.HOLD_FREQ:
            pass
        elif action == self.RAISE_FREQ:
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

        ### REWARD:
        self._reward      = self.get_reward(next_state, self._state)
        self._acc_reward += self._reward

        ### GOAL: no goal.

        ### INFO AND STATE UPDATE:
        self._info['state']     = next_state
        self._info['interval']  = self.INTERVALS[next_state - 1]

        self._info['reward']     = self._reward
        self._info['acc_reward'] = self._acc_reward

        self._info['freqpos']   = self._freqpos
        self._info['frequency'] = self._frequencies[ self._freqpos ]

        self._info['delta']     = next_power - self._power
        self._info['power']     = next_power


        self._power = next_power
        self._state = next_state
        self._count += 1

        ### RETURN:
        return [self._state, self._reward, self._done, self._info]

    def render(self, mode='human'):
        ### Print current environtment state info.
        print(self._info)

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
            # List of given points.
            powers = sorted(config['powpoints'])
        elif 'pownum' in config:
            # Number of points between MINPOWER and MAXPOWER.
            num = config['pownum']
            pstep = (self.MAXPOWER - self.MINPOWER) / (num + 1)
            ppoint = self.MINPOWER
            for _ in range(num + 2):
                powers.append(ppoint)
                ppoint += pstep
        else:
            # Step from MINPOWER until MAXPOWER is passed.
            pstep = config.get('powstep', self.DEF_POWERSTEP)
            ppoint = self.MINPOWER
            powers.append(ppoint)
            while ppoint < self.MAXPOWER:
                ppoint += pstep
                powers.append(ppoint)

        return powers

    def get_intervals(self, powerpoints):
        intervals = []

        # First interval.
        ppoint = powerpoints[0]
        intervals.append( [None, ppoint] )

        for i in range(1, len(powerpoints)):
            intervals.append( [ppoint, powerpoints[i]] )
            ppoint = powerpoints[i]

        # Last interval.
        intervals.append( [ppoint, None] )

        return intervals

    def get_state(self, power):
        pos = np.searchsorted(self.POWERPOINTS, power, side='right')

        return pos + 1

    def get_reward(self, state, prev_state):
        ### Positive while on goal.
        if state == self._goal:
            return 2

        if state < self._goal:
            if state - prev_state > 0:
                return 1
            else:
                return -1
        if state > self._goal:
            if state - prev_state < 0:
                return 1
            else:
                return -1

        ### Negative based on distance to goal.
        #return ( - abs(state - self._goal) )

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

        while meter._results is None or meter._results.pkg is None:
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
