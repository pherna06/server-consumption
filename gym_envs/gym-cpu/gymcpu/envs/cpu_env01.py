import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cpufreq
import pyRAPL
import time
import numpy as np

class CPUEnv01(gym.Env):
    ### DEFAULT PERSONAL VALUES
    DEF_POWERLIMIT = 65.0

    DEF_SOCKET = 0
    DEF_CORES  = [0,1,2,3,4,5,6,7]

    DEF_MAXSTEPS = 20
    DEF_SEED     = None
    
    DEF_MINPOWER = 15.0 - 5.0
    DEF_MAXPOWER = 115.0 + 5.0

    DEF_DECISION = 0.25 # 4 decisions / sec

    def __init__(self, **config):
        ### CPUEnv constant values.
        #   LIMIT power limit for environment functionality
        #   SOCKET socket to get pyRAPL measures
        #   CORES CPU cores assigned to SOCKET
        #   MAXSTEPS maximum iterations for environment
        #   TIME time spent in each rapl measurement
        self.LIMIT  = config.get('power',  DEF_POWERLIMIT)
        
        self.SOCKET = config.get('socket', DEF_SOCKET)
        self.CORES  = config.get('cores',  DEF_CORES)
        
        self.MAXSTEPS = config.get('maxsteps', DEF_MAXSTEPS)
        self.SEED     = config.get('seed',     DEF_SEED)

        self.MINPOWER = config.get('minpower', DEF_MINPOWER)
        self.MAXPOWER = config.get('maxpower', DEF_MAXPOWER)
        assert(self.MINPOWER < self.MAXPOWER)

        self.DECISION_TIME = config.get('decision_time', DEF_DECISION)
        self.MEASURE_TIME  = config.get('measure_time',  self.DECISION_TIME)
        self.SLEEP_TIME    = self.DECISION_TIME - self.MEASURE_TIME
        assert(self.SLEEP_TIME >= 0)

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
        #   REWARD_LOWER_BELOW lower frequency while below power limit
        #   REWARD_RAISE_BELOW raise frequency while below power limit
        #   REWARD_LOWER_ABOVE lower frequency while above power limit
        #   REWARD_RAISE_ABOVE raise frequency while above power limit
        self.REWARD_LOWER_BELOW = -1
        self.REWARD_RAISE_BELOW =  1
        self.REWARD_LOWER_ABOVE =  5
        self.REWARD_RAISE_ABOVE = -5
       
        ### Observation space:
        #   Power of CPU.
        #   _state; current power consumption
        self.observation_space = gym.spaces.Box(
                low   = self.MINPOWER,
                high  = self.MAXPOWER,
                shape = (1,),
                dtype = np.float32
                )
        
        self._state = [0.0]
        
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

        ### Decide random initial frequency.
        self._freqpos = self.RNG.choice( np.arange( len(self._frequencies) ) )
        freq = self._frequencies[ self._freqpos ]

        ### Initialize pYRAPL.
        pyRAPL.setup( 
            devices=[pyRAPL.Device.PKG], 
            socket_ids=[self.SOCKET] 
            )

        ### Set frequency, wait and measure power.
        self._state[0] = self.set_wait_measure(freq, 'Reset')

        return self._state        

    def step(self, action):        
        ### Check if max steps reached.
        if self._count == self.MAXSTEPS:
            self._done = True
            return self._state, self._reward, self._done, self._info

        assert self.action_space.contains(action)
        
        # Boolean for below/above limit.
        overlimit = True if self._state[0] > self.LIMIT else False

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
        next_state = self.set_wait_measure(freq, label)

        ### REWARDS:
        if action == self.RAISE_FREQ:
            self._reward += self.REWARD_RAISE_ABOVE if overlimit else self.REWARD_RAISE_BELOW
        elif action == self.LOWER_FREQ:
            self._reward += self.REWARD_LOWER_ABOVE if overlimit else self.REWARD_LOWER_BELOW

        ### GOAL:
        #   if overlimit, action was lower freq and new power is below limit
        if action == self.LOWER_FREQ and overlimit and next_state < self.LIMIT:
            self._done = True

        ### INFO AND STATE UPDATE:
        self._info['delta']     = next_state - self._state[0]
        self._info['power']     = next_state
        self._info['frequency'] = self._frequencies[ self._freqpos ]
        
        self._state[0] = next_state

        ### RETURN:
        self._count += 1
        return [self._state, self._reward, self._done, self._info]

    def render(self, mode='human'):
        ### Print current environtment state info.
        print(
            f"power: {self._state}, ",
            f"reward: {self._reward}, ",
            f"info: {self._info}"
        )

    def seed(self, seed=None):
        ### Make random number generator from seed.
        self.RNG, seed = seeding.np_random(seed)

        return [seed]

    def close(self):
        ### Reset CPU to default system values.
        self._cpu.reset()

    ### AUXILIARY METHODS

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
        set_frequencies(freq)

        time.sleep(self.SLEEP_TIME)

        power = measure_power(label)

        return power
