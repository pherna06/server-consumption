import numpy as np

import time
import cpufreq

########################
### NUMPY OPERATIONS ###
########################

MAX_INT = 10000000

def int_product(size):
    """
        Performs the product of random integer matrices in an infinite loop.

        Parameters
        ----------
        size : int
            The dimension of the square matrices that will be used in the
            operation.
    """
    # Random matrix generation.
    matA = np.random.randint(MAX_INT, size=(size, size))
    matB = np.random.randint(MAX_INT, size=(size, size))

    while(True):
        matC = np.matmul(matA, matB)


def float_product(size):
    """
        Performs the product of random real matrices in an infinite loop.

        Parameters
        ----------
        size : int
            The dimension of the square matrices that will be used in the
            operation.
    """
    # Random matrix generation.
    matA = np.random.rand(size, size)
    matB = np.random.rand(size, size)    

    while(True):
        matC = np.matmul(matA, matB)


def int_transpose(size):
    """
        Performs the transposition of a random integer matrix in an infinite
        loop. The numpy copy() method is used so that transposition is done 
        'physically' in memory; otherwise numpy would permorm it in constant
        time by swapping axes.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
    """
    # Random matrix generation.
    matA = np.random.randint(MAX_INT, size=(size, size))

    while(True):
        matA = matA.transpose().copy()


def float_transpose(size):
    """
        Performs the transposition of a random real matrix in an infinite
        loop. The numpy copy() method is used so that transposition is done 
        'physically' in memory; otherwise numpy would permorm it in constant
        time by swapping axes.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
    """
    # Random matrix generation.
    matA = np.random.rand(size, size)

    while(True):
        matA = matA.transpose().copy()


def int_sort(size):
    """
        Performs the sorting of a random integer array in an infinite loop.

        Parameters
        ----------
        size : int
            The squared-root value of the size of the array that will be used
            in the operation. That is, the number of elements in the array is
            size * size.
    """
    # Random array generation
    arrayA = np.random.randint(MAX_INT, size=(size*size))

    while(True):
        arrayB = np.sort(arrayA)


def float_sort(size):
    """
        Performs the sorting of a random real array in an infinite loop.

        Parameters
        ----------
        size : int
            The squared-root value of the size of the array that will be used
            in the operation. That is, the number of elements in the array is
            size * size.
    """
    # Random array generation
    arrayA = np.random.rand(size*size)

    while(True):
        arrayB = np.sort(arrayA)


def int_scalar(size):
    """
        Performs the sum of a random integer to each element of a random 
        integer matrix in an infinite loop.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
    """
    # Random matrix generation
    matA = np.random.randint(MAX_INT, size=(size, size))
    intN = np.random.randint(MAX_INT)

    while(True):
        matB = matA + intN

def float_scalar(size):
    """
        Performs the sum of a random real number to each element of a random 
        real matrix in an infinite loop.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
    """
    # Random matrix generation
    matA = np.random.rand(size, size)
    floatN = np.random.rand()

    while(True):
        matB = matA + floatN



OPERATIONS = {
    'intproduct':       int_product,
    'inttranspose':     int_transpose,
    'intsort':          int_sort,
    'intscalar':        int_scalar,
    'floatproduct':     float_product,
    'floattranspose':   float_transpose,
    'floatsort':        float_sort,
    'floatscalar':      float_scalar
}

#########################
### CPUFREQ UTILITIES ###
#########################

_cpu = cpufreq.cpuFreq()
_available_freqs = sorted(_cpu.available_frequencies)

def closest_frequency(freq):
    """
        Approximates the specified frequency to the closest higher or equal
        available frequency.

        Parameters
        ----------
        freq : int
            The specified frequency, in KHz, to be approximated. Must not be
            smaller than the minimum available frequency.

        Returns 
        -------
        int
            The minimum frequency of _available_freqs which is higher or equal
            than the given frequency.
    """
    if freq < _available_freqs[0]:
        print("ERROR: Specified frequency is below the minimum allowed frequency.")
        exit()

    av_freq = _available_freqs[0]
    for af in _available_freqs:
        if freq <= av_freq:
            break
        av_freq = af

    return av_freq


def lower_frequency(freq, rg):
    """
        Reduces the frequency of the desired online CPU cores to the specified
        frequency. Frequency must be lower than current frequency; otherwise
        an error will occur.

        Parameters
        ----------
        freq : int
            The specified frequency, in KHz. Must be lower than current 
            frequencies of affected CPU cores.
        rg : int, list
            An integer or list of integers with the indices of CPU cores whose
            frequency will be lowered. If None, all online CPU cores will be
            modified.
    """
    _cpu.set_min_frequencies(freq, rg)
    _cpu.set_max_frequencies(freq, rg)
    _cpu.set_frequencies(freq, rg)

def raise_frequency(freq, rg):
    """
        Increases the frequency of the desired online CPU cores to the 
        specified frequency. Frequency must be higher than current frequency; 
        otherwise an error will occur.

        Parameters
        ----------
        freq : int
            The specified frequency, in KHz. Must be higher than current 
            frequencies of affected CPU cores.
        rg : int, list
            An integer or list of integers with the indices of CPU cores whose
            frequency will be raised. If None, all online CPU cores will be
            modified.
    """
    _cpu.set_max_frequencies(freq, rg)
    _cpu.set_min_frequencies(freq, rg)
    _cpu.set_frequencies(freq, rg)

