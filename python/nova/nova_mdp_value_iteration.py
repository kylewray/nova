""" The MIT License (MIT)

    Copyright (c) 2016 Kyle Hollins Wray, University of Massachusetts

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
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

import mdp
import mdp_value_function as mvf

# Check if we need to create the nova variable. If so, import the correct library
# file depending on the platform.
#try:
#    _nova
#except NameError:
_nova = None
if platform.system() == "Windows":
    _nova = ct.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "..", "..", "lib", "libnova.dll"))
else:
    _nova = ct.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "..", "..", "lib", "libnova.so"))


class NovaMDPValueIterationCPU(ct.Structure):
    """ The C struct MDPValueIterationCPU object. """

    _fields_ = [("Vinitial", ct.POINTER(ct.c_float)),
                ("currentHorizon", ct.c_uint),
                ("V", ct.POINTER(ct.c_float)),
                ("Vprime", ct.POINTER(ct.c_float)),
                ("pi", ct.POINTER(ct.c_uint)),
                ]


_nova.mdp_vi_initialize_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                    ct.POINTER(NovaMDPValueIterationCPU))

_nova.mdp_vi_execute_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                    ct.POINTER(NovaMDPValueIterationCPU),
                                    ct.POINTER(ct.POINTER(mvf.MDPValueFunction)))

_nova.mdp_vi_uninitialize_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                    ct.POINTER(NovaMDPValueIterationCPU))

_nova.mdp_vi_update_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                    ct.POINTER(NovaMDPValueIterationCPU))

_nova.mdp_vi_get_policy_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                        ct.POINTER(NovaMDPValueIterationCPU),
                                        ct.POINTER(ct.POINTER(mvf.MDPValueFunction)))


class NovaMDPValueIterationGPU(ct.Structure):
    """ The C struct MDPValueIterationGPU object. """

    _fields_ = [("Vinitial", ct.POINTER(ct.c_float)),
                ("numThreads", ct.c_uint),
                ("currentHorizon", ct.c_uint),
                ("d_V", ct.POINTER(ct.c_float)),
                ("d_Vprime", ct.POINTER(ct.c_float)),
                ("d_pi", ct.POINTER(ct.c_uint)),
                ]


_nova.mdp_vi_initialize_gpu.argtypes = (ct.POINTER(mdp.MDP),
                                    ct.POINTER(NovaMDPValueIterationGPU))

_nova.mdp_vi_execute_gpu.argtypes = (ct.POINTER(mdp.MDP),
                                    ct.POINTER(NovaMDPValueIterationGPU),
                                    ct.POINTER(ct.POINTER(mvf.MDPValueFunction)))

_nova.mdp_vi_uninitialize_gpu.argtypes = (ct.POINTER(mdp.MDP),
                                    ct.POINTER(NovaMDPValueIterationGPU))

_nova.mdp_vi_update_gpu.argtypes = (ct.POINTER(mdp.MDP),
                                    ct.POINTER(NovaMDPValueIterationGPU))

_nova.mdp_vi_get_policy_gpu.argtypes = (ct.POINTER(mdp.MDP),
                                        ct.POINTER(NovaMDPValueIterationCPU),
                                        ct.POINTER(ct.POINTER(mvf.MDPValueFunction)))

