import numpy as np
import argparse
import os

# MODIFY ACCORDING TO YOUR MACHINE CPU CONFIGURATION.
CPU_LIST = list(range(16))
SOCKET_LIST = [0, 1]
SOCKET_DICT = {
        0: CPU_LIST[0:8], 
        1: CPU_LIST[8:16]
}

# DIMENSION
DEF_DIM = 1000

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
        

#########################
### COMMAND INTERFACE ###
#########################

def get_parser():
    desc = ""
    parser = argparse.ArgumentParser(description = desc)

    ## Execution: cores or sockets
    cpucores = parser.add_mutually_exclusive_group()

    cores_help = "The operation is executed in each specified core."
    cpucores.add_argument(
        '-c', '--cores', metavar='cores', help=cores_help,
        nargs='+',
        type=int,
    )

    sockets_help = "The operation is executed in the cores of the specified "
    sockets_help += "sockets."
    cpucores.add_argument(
        '-s', '--sockets', metavar='sockets', help=sockets_help,
        nargs='+',
        type=int
    )

    # Optional arguments
    ## Dimension
    dim_help = "Dimension used for the Numpy elements in the operation."
    dim_help += "For matrices, size is DIM X DIM."
    dim_help += "Default value for DIM is {}".format(DEF_DIM)
    parser.add_argument(
        '-d', '--dim', metavar='dim', help=dim_help, 
        type=int, 
        default=DEF_DIM
    )
  
    # Positional arguments.
    work_help = "Name of operation to be tested: intproduct inttranspose "
    work_help += "intsort intscalar floatproduct floattranspose floatsort "
    work_help += "floatscalar"
    parser.add_argument('work', help=work_help)

    return parser

def get_cores(sockets):
    rg = []
    for skt in sockets:
        rg.extend(SOCKET_DICT[skt])

    return rg

def main():
    """
        Given the name of an operation, it is executed with the specified
        affinity and dimension. As these operations do not return, this process
        must be closed externally.
    """
    parser = get_parser()
    args = parser.parse_args()

    # Set operation affinity
    if 'affcores' in args:
        os.sched_setaffinity(0, args.affcores)
    elif 'addsockets' in args:
        os.sched_setaffinity(0, get_cores(args.affsockets))

    # Show PID:
    print("PID:", os.getpid())

    OPERATIONS[args.work](size=args.dim)

if __name__ == '__main__':
    main()