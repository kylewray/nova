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
import nova_pomdp as npm


def pomdp_pbvi_complete_cpu(n, ns, m, z, r, rz, gamma, horizon,
        S, T, O, R, Z, B, numThreads, Gamma, pi):
    """ The wrapper Python function for executing point-based value iteration for a POMDP using the CPU.

        Parameters:
            n                   --  The number of states.
            ns                  --  The maximum number of successor states.
            m                   --  The number of actions.
            z                   --  The number of observations.
            r                   --  The number of belief points.
            rz                  --  The maximum number of non-zero belief values over all beliefs.
            gamma               --  The discount factor.
            horizon             --  The number of iterations.
            S                   --  The state-action pairs as a flattened 2-dimensional array.
            T                   --  The state transitions as a flattened 3-dimensional array.
            O                   --  The observation transitions as a flattened 3-dimensional array.
            R                   --  The reward function as a flattened 2-dimensional array.
            Z                   --  The belief-state pairs as a flattened 2-dimensional array.
            B                   --  The belief points as a flattened 2-dimensional array.
            Gamma               --  The resultant alpha-vectors. Modified.
            pi                  --  The resultant actions for each alpha-vector. Modified.

        Returns:
            Zero on success, and a non-zero nova error code otherwise.
    """

    array_type_nmns_int = ct.c_int * (int(n) * int(m) * int(ns))
    array_type_nmns_float = ct.c_float * (int(n) * int(m) * int(ns))
    array_type_mnz_float = ct.c_float * (int(m) * int(n) * int(z))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_rrz_int = ct.c_int * (int(r) * int(rz))
    array_type_rrz_float = ct.c_float * (int(r) * int(rz))

    array_type_rn_float = ct.c_float * (int(r) * int(n))
    array_type_r_uint = ct.c_uint * int(r)

    GammaResult = array_type_rn_float(*Gamma)
    piResult = array_type_r_uint(*pi)

    result = npm._nova.pomdp_pbvi_complete_cpu(npm.NovaPOMDP(int(n), int(ns), int(m), int(z),
                                int(r), int(rz), float(gamma), int(horizon),
                                array_type_nmns_int(*S), array_type_nmns_float(*T),
                                array_type_mnz_float(*O), array_type_nm_float(*R),
                                array_type_rrz_int(*Z), array_type_rrz_float(*B),
                                int(0),
                                ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)(),
                                ct.POINTER(ct.c_int)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_int)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)(),
                                ct.POINTER(ct.c_float)()),
                            GammaResult, piResult)

    if result == 0:
        for i in range(r * n):
            Gamma[i] = GammaResult[i]
        for i in range(r):
            pi[i] = piResult[i]

    return result

