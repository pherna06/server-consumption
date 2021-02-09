import numpy as np

import time

####################
### CALL UTILITY ###
####################

def timeop(op, config):
    """
        Utility function to select the operation function within this module
        externally.
        
        Parameters
        ----------
        op : str
            Name associated with the operation to be measured.
        config : dict
            Dictionary of arguments (size) for the operation function.

        Returns
        -------
        float
            The execution time associated with the selected operation.
    """
    return OPERATIONS.get(op, lambda: 0.0)(**config)

########################
### NUMPY OPERATIONS ###
########################

# DICT RELATION
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

# DEFAULT RANDOM MATRICES SIZE.
DEF_MATRIX = 1000
DEF_LIST = DEF_MATRIX * DEF_MATRIX

# NUMBER GENERATION
DEF_MAXINT = 1000000

# DEFAULT ITERATIONS
DEF_REP = 1

def int_product(size=DEF_MATRIX, rep=DEF_REP):
    """
        Returns the execution time of performing a product of random integer
        matrices.

        Parameters
        ----------
        size : int
            The dimension of the square matrices that will be used in the
            operation.
        rep : int
            Number of times the operation will be performed.
        
        Returns
        -------
        float
            The mean execution time of the matrix product.
    """
    acc = 0
    for _ in range(0, rep):
        # Random matrix generation.
        matA = np.random.randint(DEF_MAXINT, size=(size, size))
        matB = np.random.randint(DEF_MAXINT, size=(size, size))

        # TIME: operation.
        start = time.time()
        matC = np.matmul(matA, matB)
        end = time.time()

        acc += (end - start)

    return (acc / rep)

def float_product(size=DEF_MATRIX, rep=DEF_REP):
    """
        Returns the execution time of performing a product of random real
        matrices.

        Parameters
        ----------
        size : int
            The dimension of the square matrices that will be used in the
            operation.
        rep : int
            Number of times the operation will be performed.
        
        Returns
        -------
        float
            The mean execution time of the matrix product.
    """
    acc = 0
    for _ in range(0, rep):
        # Random matrix generation.
        matA = np.random.rand(size, size)
        matB = np.random.rand(size, size)    

        # TIME: operation.
        start = time.time()
        matC = np.matmul(matA, matB)
        end = time.time()

        acc += (end - start)

    return (acc / rep)

def int_transpose(size=DEF_MATRIX, rep=DEF_REP):
    """
        Return the execution time of performing the transposition of an integer
        matrix twice. The numpy copy() method is used so that transposition is
        done 'physically' in memory; otherwise numpy would permorm it in constant
        time by swapping axes.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
        rep : int
            Number of times the operation will be performed.
        
        Returns
        -------
        float
            The mean execution time of the transposition.
    """
    acc = 0
    for _ in range(0, rep):
        # Random matrix generation.
        matA = np.random.randint(DEF_MAXINT, size=(size, size))

        # TIME: operation.
        start = time.time()
        matA = matA.transpose().copy()
        matA = matA.transpose().copy()
        end = time.time()

        acc += (end - start)

    return (acc / rep)

def float_transpose(size=DEF_MATRIX, rep=DEF_REP):
    """
        Return the execution time of performing the transposition of an real
        matrix twice. The numpy copy() method is used so that transposition is
        done 'physically' in memory; otherwise numpy would permorm it in constant
        time by swapping axes.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
        rep : int
            Number of times the operation will be performed.
        
        Returns
        -------
        float
            The mean execution time of the transposition.
    """
    acc = 0
    for _ in range(0, rep):
        # Random matrix generation.
        matA = np.random.rand(size, size)

        # TIME: operation.
        start = time.time()
        matA = matA.transpose().copy()
        matA = matA.transpose().copy()
        end = time.time()

        acc += (end - start)

    return (acc / rep)

def int_sort(size=DEF_LIST, rep=DEF_REP):
    """
        Returns the execution time of sorting a random integer array.

        Parameters
        ----------
        size : int
            The squared-root value of the size of the array that will be used
            in the operation. That is, the number of elements in the array is
            size * size.
        rep : int
            Number of times the operation will be performed.
        
        Returns
        -------
        float
            The mean execution time of sorting the array.
    """
    acc = 0
    for _ in range(0, rep):
        # Random array generation
        arrayA = np.random.randint(DEF_MAXINT, size=(size*size))

        # TIME: operation
        start = time.time()
        arrayB = np.sort(arrayA)
        end = time.time()

        acc += (end - start)

    return (acc / rep)

def float_sort(size=DEF_LIST, rep=DEF_REP):
    """
        Returns the execution time of sorting a random real array.

        Parameters
        ----------
        size : int
            The squared-root value of the size of the array that will be used
            in the operation. That is, the number of elements in the array is
            size * size.
        rep : int
            Number of times the operation will be performed.
        
        Returns
        -------
        float
            The mean execution time of sorting the array.
    """
    acc = 0
    for _ in range(0, rep):
        # Random array generation
        arrayA = np.random.rand(size*size)

        # TIME: operation
        start = time.time()
        arrayB = np.sort(arrayA)
        end = time.time()

        acc += (end - start)

    return (acc / rep)

def int_scalar(size=DEF_MATRIX, rep=DEF_REP):
    """
        Returns the execution time of performing the sum of a random integer to
        each element of a random integer matrix.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
        rep : int
            Number of times the operation will be performed.

        Returns
        -------
        float
            The mean execution time of sorting the array.
    """
    acc = 0
    for _ in range(0, rep):
        # Random matrix generation
        matA = np.random.randint(DEF_MAXINT, size=(size, size))
        intN = np.random.randint(DEF_MAXINT)

        # TIME: operation.
        start = time.time()
        matB = matA + intN
        end = time.time()

        acc += (end - start)

    return (acc / rep)

def float_scalar(size=DEF_MATRIX, rep=DEF_REP):
    """
        Returns the execution time of performing the sum of a random number to
        each element of a random real matrix.

        Parameters
        ----------
        size : int
            The dimension of the square matrix that will be used in the 
            operation.
        rep : int
            Number of times the operation will be performed.

        Returns
        -------
        float
            The mean execution time of sorting the array.
    """
    acc = 0
    for _ in range(0, rep):
        # Random matrix generation
        matA = np.random.rand(size, size)
        floatN = np.random.rand()

        # TIME: operation.
        start = time.time()
        matB = matA + floatN
        end = time.time()

        acc += (end - start)

    return (acc / rep)