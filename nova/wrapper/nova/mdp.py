""" The MIT License (MIT)

    Copyright (c) 2015 Kyle Hollins Wray, University of Massachusetts

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import ctypes as ct
import platform
import os.path

# Import the correct library file depending on the platform.
_nova = None
if platform.system() == "Windows":
    _nova = ct.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "..", "..", "lib", "nova.dll"))
else:
    _nova = ct.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "..", "..", "lib", "nova.so"))

_nova.nova_mdp_vi.argtypes = (ct.c_uint, # n
                                ct.c_uint, # m
                                ct.POINTER(ct.c_float), # T
                                ct.POINTER(ct.c_float), # R
                                ct.c_float, # gamma
                                ct.c_uint, # horizon
                                ct.c_uint, # numThreads
                                ct.POINTER(ct.c_float), # V
                                ct.POINTER(ct.c_uint)) # pi

def nova_mdp_vi(n, m, T, R, gamma, horizon, numThreads, V, pi):
    """ The wrapper Python function for executing value iteration for an MDP.

        Parameters:
            n           --  The number of states.
            m           --  The number of actions.
            T           --  The state transitions as a flattened 3-dimensional array.
            R           --  The reward function as a flattened 2-dimensional array.
            gamma       --  The discount factor.
            horizon     --  The number of iterations to execute.
            numThreads  --  The number of CUDA threads to execute.
            V           --  The resultant values of the states. Modified.
            pi          --  The resultant actions to take at each state. Modified.

        Returns:
            0   --  Success.
            1   --  Invalid arguments were passed.
            2   --  The number of blocks and threads is less than the number of states.
            3   --  A CUDA memcpy failed somewhere, which should also output to std::err.
    """

    global _nova

    array_type_nmn_float = ct.c_float * (int(n) * int(m) * int(n))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_n_float = ct.c_float * int(n)
    array_type_n_uint = ct.c_uint * int(n)

    VResult = array_type_n_float(*V)
    piResult = array_type_n_uint(*pi)

    result = _nova.nova_mdp_vi(int(n), int(m), array_type_nmn_float(*T), array_type_nm_float(*R),
                            float(gamma), int(horizon), int(numThreads), VResult, piResult)

    if result == 0:
        for i in range(n):
            V[i] = VResult[i]
            pi[i] = piResult[i]

    return result

