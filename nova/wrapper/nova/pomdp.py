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

_nova.nova_pomdp_pbvi.argtypes = (ct.c_uint, # n
                                ct.c_uint, # m
                                ct.c_uint, # z
                                ct.c_uint, # r
                                ct.c_uint, # maxNonZeroBeliefs
                                ct.c_uint, # maxSuccessors
                                ct.POINTER(ct.c_float), # B
                                ct.POINTER(ct.c_float), # T
                                ct.POINTER(ct.c_float), # O
                                ct.POINTER(ct.c_float), # R
                                ct.POINTER(ct.c_bool), # available
                                ct.POINTER(ct.c_int), # nonZeroBeliefs
                                ct.POINTER(ct.c_int), # successors
                                ct.c_float, # gamma
                                ct.c_uint, # horizon
                                ct.c_uint, # numThreads
                                ct.POINTER(ct.c_float), # Gamma
                                ct.POINTER(ct.c_uint)) # pi

def nova_pomdp_pbvi(n, m, z, r, maxNonZeroBeliefs, maxSuccessors,
        B, T, O, R, available, successors,
        gamma, horizon, numThreads, Gamma, pi):
    """ The wrapper Python function for executing point-based value iteration for a POMDP.

        Parameters:
            n                   --  The number of states.
            m                   --  The number of actions.
            z                   --  The number of observations.
            r                   --  The number of belief points.
            maxNonZeroBeliefs   --  The maximum number of non-zero belief values over all beliefs.
            maxSuccessors       --  The maximum number of successor states.
            B                   --  The belief points as a flattened 2-dimensional array.
            T                   --  The state transitions as a flattened 3-dimensional array.
            O                   --  The observation transitions as a flattened 3-dimensional array.
            R                   --  The reward function as a flattened 2-dimensional array.
            available           --  The belief-action pairs as a flattened 2-dimensional array.
            nonZeroBeliefs      --  The belief-state pairs as a flattened 2-dimensional array.
            successors          --  The state-action pairs as a flattened 2-dimensional array.
            gamma               --  The discount factor.
            horizon             --  The number of iterations.
            numThreads          --  The number of CUDA threads to execute.
            Gamma               --  The resultant alpha-vectors. Modified.
            pi                  --  The resultant actions for each alpha-vector. Modified.

        Returns:
            0   --  Success.
            1   --  Invalid arguments were passed.
            2   --  The number of blocks and threads is less than the number of states.
            3   --  A CUDA memcpy failed somewhere, which should also output to std::err.
    """

    global _nova

    array_type_rn_float = ct.c_float * (int(r) * int(n))
    array_type_nmn_float = ct.c_float * (int(n) * int(m) * int(n))
    array_type_mnz_float = ct.c_float * (int(m) * int(n) * int(z))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_rm_bool = ct.c_bool * (int(r) * int(m))
    array_type_rxb_int = ct.c_int * (int(r) * int(maxNonZeroBeliefs))
    array_type_nmxs_int = ct.c_int * (int(n) * int(m) * int(maxSuccessors))
    array_type_r_uint = ct.c_uint * int(r)

    GammaResult = array_type_rm_float(*Gamma)
    piResult = array_type_r_uint(*pi)

    result = _nova.nova_pomdp_pbvi(int(n), int(m), int(z), int(r), int(maxNonZeroBeliefs),
                            int(maxSuccessors), array_type_rn_float(*B),
                            array_type_nmn_float(*T), array_type_mnz_float(*O),
                            array_type_nm_float(*R), array_type_rm_bool(*available),
                            array_type_rxb_int(*nonZeroBeliefs), array_type_nmxs_int(*successors),
                            float(gamma), int(horizon), int(numThreads),
                            GammaResult, piResult)

    if result == 0:
        for i in range(r * n):
            Gamma[i] = GammaResult[i]
        for i in range(r):
            pi[i] = piResult[i]

    return result

