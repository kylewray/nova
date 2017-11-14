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
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
import nova_pomdp_alpha_vectors as npav
import pomdp_alpha_vectors as pav


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


class NovaPOMDP(ct.Structure):
    """ The C struct POMDP object. """

    _fields_ = [("n", ct.c_uint),
                ("ns", ct.c_uint),
                ("m", ct.c_uint),
                ("z", ct.c_uint),
                ("r", ct.c_uint),
                ("rz", ct.c_uint),
                ("gamma", ct.c_float),
                ("horizon", ct.c_uint),
                ("S", ct.POINTER(ct.c_int)),
                ("T", ct.POINTER(ct.c_float)),
                ("O", ct.POINTER(ct.c_float)),
                ("R", ct.POINTER(ct.c_float)),
                ("Z", ct.POINTER(ct.c_int)),
                ("B", ct.POINTER(ct.c_float)),
                ("d_S", ct.POINTER(ct.c_int)),
                ("d_T", ct.POINTER(ct.c_float)),
                ("d_O", ct.POINTER(ct.c_float)),
                ("d_R", ct.POINTER(ct.c_float)),
                ("d_Z", ct.POINTER(ct.c_int)),
                ("d_B", ct.POINTER(ct.c_float)),
                ]


# Functions from 'pomdp_model.h'.
_nova.pomdp_initialize.argtypes = (ct.POINTER(NovaPOMDP),
                                   ct.c_uint,   # n
                                   ct.c_uint,   # ns
                                   ct.c_uint,   # m
                                   ct.c_uint,   # z
                                   ct.c_uint,   # r
                                   ct.c_uint,   # rz
                                   ct.c_float,  # gamma
                                   ct.c_uint)   # horizon
_nova.pomdp_belief_update.argtypes = (ct.POINTER(NovaPOMDP),
                                      ct.POINTER(ct.c_float),                 # b
                                      ct.c_uint,                              # a
                                      ct.c_uint,                              # o
                                      ct.POINTER(ct.POINTER(ct.c_float)))     # bp
_nova.pomdp_add_new_raw_beliefs.argtypes = (ct.POINTER(NovaPOMDP),
                                            ct.c_uint,                  # numBeliefPointsToAdd
                                            ct.POINTER(ct.c_float))     # Bnew
_nova.pomdp_uninitialize.argtypes = tuple([ct.POINTER(NovaPOMDP)])


# Functions from 'pomdp_expand.h'.
_nova.pomdp_expand_random.argtypes = (ct.POINTER(NovaPOMDP),
                                      ct.c_uint)    # numBeliefsToAdd
_nova.pomdp_expand_random_unique.argtypes = (ct.POINTER(NovaPOMDP),
                                             ct.c_uint,    # numBeliefsToAdd
                                             ct.c_uint)    # maxTrials
_nova.pomdp_expand_distinct_beliefs.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_expand_pema.argtypes = (ct.POINTER(NovaPOMDP),
                                    ct.POINTER(pav.POMDPAlphaVectors))  # policy


# Functions from 'pomdp_sigma.h'.
_nova.pomdp_sigma.argtypes = (ct.POINTER(NovaPOMDP),
                              ct.c_uint,                # numDesiredNonZeroValues
                              ct.POINTER(ct.c_float))   # sigma


# Functions from 'pomdp_model_gpu.h'.
_nova.pomdp_initialize_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_uninitialize_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_initialize_successors_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_uninitialize_successors_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_initialize_state_transitions_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_uninitialize_state_transitions_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_initialize_observation_transitions_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_uninitialize_observation_transitions_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_initialize_rewards_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_uninitialize_rewards_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_initialize_nonzero_beliefs_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_uninitialize_nonzero_beliefs_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_initialize_belief_points_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_uninitialize_belief_points_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])


# Functions from 'pomdp_expand_gpu.h'.
_nova.pomdp_expand_random_gpu.argtypes = (ct.POINTER(NovaPOMDP),
                                          ct.c_uint,  # numThreads
                                          ct.c_uint)  # numBeliefsToAdd

