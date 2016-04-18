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


class NovaMDP(ct.Structure):
    """ The C struct MDP object. """

    _fields_ = [("n", ct.c_uint),
                ("ns", ct.c_uint),
                ("m", ct.c_uint),
                ("gamma", ct.c_float),
                ("horizon", ct.c_uint),
                ("epsilon", ct.c_float),
                ("s0", ct.c_uint),
                ("ng", ct.c_uint),
                ("goals", ct.POINTER(ct.c_uint)),
                ("S", ct.POINTER(ct.c_int)),
                ("T", ct.POINTER(ct.c_float)),
                ("R", ct.POINTER(ct.c_float)),
                ("d_goals", ct.POINTER(ct.c_uint)),
                ("d_S", ct.POINTER(ct.c_int)),
                ("d_T", ct.POINTER(ct.c_float)),
                ("d_R", ct.POINTER(ct.c_float)),
                ]


# Functions from 'mdp_model_gpu.h'.
_nova.mdp_uninitialize_cpu.argtypes = tuple([ct.POINTER(NovaMDP)])


# Functions from 'mdp_model_gpu.h'.
_nova.mdp_initialize_successors_gpu.argtypes = tuple([ct.POINTER(NovaMDP)])
_nova.mdp_uninitialize_successors_gpu.argtypes = tuple([ct.POINTER(NovaMDP)])
_nova.mdp_initialize_state_transitions_gpu.argtypes = tuple([ct.POINTER(NovaMDP)])
_nova.mdp_uninitialize_state_transitions_gpu.argtypes = tuple([ct.POINTER(NovaMDP)])
_nova.mdp_initialize_rewards_gpu.argtypes = tuple([ct.POINTER(NovaMDP)])
_nova.mdp_uninitialize_rewards_gpu.argtypes = tuple([ct.POINTER(NovaMDP)])
_nova.mdp_initialize_goals_gpu.argtypes = tuple([ct.POINTER(NovaMDP)])
_nova.mdp_uninitialize_goals_gpu.argtypes = tuple([ct.POINTER(NovaMDP)])

