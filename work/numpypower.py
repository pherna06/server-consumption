import numpy as np

####################
### CALL UTILITY ###
####################

def powerop(op, config):
    """
        Utility function to select the operation function within this module
        externally.
        
        Parameters
        ----------
        op : str
            Name associated with the operation to be measured.
        config : dict
            Dictionary of arguments (size) for the operation function.
    """
    OPERATIONS.get(op, lambda: None)(**config)

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
MATRIX_SIZE = 1000
LIST_SIZE = MATRIX_SIZE*MATRIX_SIZE

# NUMBER GENERATION
MAX_INT = 1000000

def int_product(size=MATRIX_SIZE):
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


def float_product(size=MATRIX_SIZE):
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


def int_transpose(size=MATRIX_SIZE):
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


def float_transpose(size=MATRIX_SIZE):
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


def int_sort(size=LIST_SIZE):
    """
        Performs the sorting of a random integer array in an infinite loop.

        Parameters
        ----------
        size : int
            The size of the array that will be used in the operation.
    """
    # Random array generation
    arrayA = np.random.randint(MAX_INT, size=(size))

    while(True):
        arrayB = np.sort(arrayA)


def float_sort(size=LIST_SIZE):
    """
        Performs the sorting of a random real array in an infinite loop.

        Parameters
        ----------
        size : int
            The size of the array that will be used in the operation.
    """
    # Random array generation
    arrayA = np.random.rand(size)

    while(True):
        arrayB = np.sort(arrayA)


def int_scalar(size=MATRIX_SIZE):
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

def float_scalar(size=MATRIX_SIZE):
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