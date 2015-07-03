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
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
import nova_mdp as nm


def mdp_vi_complete_cpu(n, ns, m, gamma, horizon, S, T, R, V, pi):
    """ The wrapper Python function for executing value iteration for an MDP using the CPU.

        Parameters:
            n           --  The number of states.
            ns          --  The maximum number of successor states.
            m           --  The number of actions.
            gamma       --  The discount factor.
            horizon     --  The number of iterations to execute.
            S           --  The successor states as a flattened 3-dimensional array (n-m-ns-array).
            T           --  The state transitions as a flattened 3-dimensional array (n-m-ns-array).
            R           --  The reward function as a flattened 2-dimensional array (n-m-array).
            V           --  The resultant values of the states (n-array). Modified.
            pi          --  The resultant actions to take at each state (n-array). Modified.

        Returns:
            Zero on success; non-zero otherwise.
    """

    array_type_nmns_int = ct.c_int * (int(n) * int(m) * int(ns))
    array_type_nmns_float = ct.c_float * (int(n) * int(m) * int(ns))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_n_float = ct.c_float * int(n)
    array_type_n_uint = ct.c_uint * int(n)

    VResult = array_type_n_float(*V)
    piResult = array_type_n_uint(*pi)

    # Note: The 'ct.POINTER(ct.c_xyz)()' below simply assigns a nullptr value in the struct
    # to the corresponding value. This device pointer will never be assigned for the CPU version.
    result = nm._nova.mdp_vi_complete_cpu(nm.NovaMDP(int(n), int(ns), int(m),
                                float(gamma), int(horizon),
                                array_type_nmns_int(*S),
                                array_type_nmns_float(*T),
                                array_type_nm_float(*R),
                                int(0),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)(),
                                ct.POINTER(ct.c_int)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)()),
                            VResult, piResult)

    if result == 0:
        for i in range(n):
            V[i] = VResult[i]
            pi[i] = piResult[i]

    return result

