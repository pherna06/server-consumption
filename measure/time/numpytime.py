import numpy as np

import time
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

    ## Repetitions
    rep_help = "Number of iterations of the selected operation to obtain mean "
    rep_help += "execution time."
    rep_help += "Default value is {}".format(DEF_REP)
    parser.add_argument(
        '-r', '--rep', metavar='rep', help='rep_help',
        type=int,
        default=DEF_REP
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
        affinity, dimension and repetitions. Execution time is printed after
        operation termination.
    """
    parser = get_parser()
    args = parser.parse_args()

    # Set operation affinity
    if 'affcores' in args:
        os.sched_setaffinity(0, args.affcores)
    elif 'addsockets' in args:
        os.sched_setaffinity(0, get_cores(args.affsockets))

    timems = OPERATIONS[args.work](size=args.dim, rep=args.rep) * 1000
    print("Mean execution time:", timems, "ms")

if __name__ == '__main__':
    main()