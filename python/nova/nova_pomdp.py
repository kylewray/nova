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
                    "..", "..", "lib", "nova.dll"))
else:
    # Linux lib name start with lib: this is just to keep backward compatibility
    lib = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "lib", "nova.so")
    if os.path.exists(lib):
        _nova = ct.CDLL(lib)
    else:
        _nova = ct.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "lib", "libnova.so"))


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
                ("currentHorizon", ct.c_uint),
                ("rGamma", ct.c_uint),
                ("rGammaPrime", ct.c_uint),
                ("BTilde", ct.POINTER(ct.c_uint)),
                ("rTilde", ct.c_uint),
                ("Gamma", ct.POINTER(ct.c_float)),
                ("GammaPrime", ct.POINTER(ct.c_float)),
                ("pi", ct.POINTER(ct.c_uint)),
                ("piPrime", ct.POINTER(ct.c_uint)),
                ("d_S", ct.POINTER(ct.c_int)),
                ("d_T", ct.POINTER(ct.c_float)),
                ("d_O", ct.POINTER(ct.c_float)),
                ("d_R", ct.POINTER(ct.c_float)),
                ("d_Z", ct.POINTER(ct.c_int)),
                ("d_B", ct.POINTER(ct.c_float)),
                ("d_Gamma", ct.POINTER(ct.c_float)),
                ("d_GammaPrime", ct.POINTER(ct.c_float)),
                ("d_pi", ct.POINTER(ct.c_uint)),
                ("d_piPrime", ct.POINTER(ct.c_uint)),
                ("d_alphaBA", ct.POINTER(ct.c_float)),
                ]


# Functions from 'pomdp_pbvi_cpu.h'.
_nova.pomdp_pbvi_complete_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                        ct.POINTER(ct.c_float),                 # initialGamma
                                        ct.POINTER(ct.POINTER(pav.POMDPAlphaVectors)))  # policy
_nova.pomdp_pbvi_initialize_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                            ct.POINTER(ct.c_float))    # initialGamma
_nova.pomdp_pbvi_execute_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                        ct.POINTER(ct.c_float),                 # initialGamma
                                        ct.POINTER(ct.POINTER(pav.POMDPAlphaVectors)))  # policy
_nova.pomdp_pbvi_uninitialize_cpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_pbvi_update_cpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_pbvi_get_policy_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                        ct.POINTER(ct.POINTER(pav.POMDPAlphaVectors)))  # policy


# Functions from 'pomdp_perseus_cpu.h'.
_nova.pomdp_perseus_complete_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                            ct.POINTER(ct.c_float),                         # initialGamma
                                            ct.POINTER(ct.POINTER(pav.POMDPAlphaVectors)))  # policy
_nova.pomdp_perseus_initialize_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                                ct.POINTER(ct.c_float))    # initialGamma
_nova.pomdp_perseus_execute_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                            ct.POINTER(ct.c_float),                         # initialGamma
                                            ct.POINTER(ct.POINTER(pav.POMDPAlphaVectors)))  # policy
_nova.pomdp_perseus_uninitialize_cpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_perseus_update_cpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_perseus_get_policy_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                            ct.POINTER(ct.POINTER(pav.POMDPAlphaVectors)))  # policy


# Functions from 'pomdp_expand_cpu.h'.
_nova.pomdp_expand_random_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                            ct.c_uint,              # numDesiredBeliefPoints
                                            ct.POINTER(ct.c_uint),  # maxNonZeroValues
                                            ct.POINTER(ct.c_float)) # Bnew
_nova.pomdp_expand_distinct_beliefs_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                                    ct.POINTER(ct.c_uint),  # maxNonZeroValues
                                                    ct.POINTER(ct.c_float)) # Bnew
_nova.pomdp_expand_pema_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                        ct.POINTER(pav.POMDPAlphaVectors), # Gamma
                                        ct.POINTER(ct.c_uint),  # maxNonZeroValues
                                        ct.POINTER(ct.c_float)) # Bnew


# Functions from 'pomdp_sigma_cpu.h'.
_nova.pomdp_sigma_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                  ct.c_uint,                # rz (the new desired one)
                                  ct.POINTER(ct.c_float),   # Bnew
                                  ct.POINTER(ct.c_int),     # Znew
                                  ct.POINTER(ct.c_float))   # sigma


# Functions from 'pomdp_model_gpu.h'.
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


# Functions from 'pomdp_pbvi_gpu.h'.
_nova.pomdp_pbvi_complete_gpu.argtypes = (ct.POINTER(NovaPOMDP),
                                        ct.c_uint,                              # numThreads
                                        ct.POINTER(ct.c_float),                 # initialGamma
                                        ct.POINTER(ct.POINTER(pav.POMDPAlphaVectors)))  # policy
_nova.pomdp_pbvi_initialize_gpu.argtypes = (ct.POINTER(NovaPOMDP),
                                            ct.POINTER(ct.c_float))    # initialGamma
_nova.pomdp_pbvi_execute_gpu.argtypes = (ct.POINTER(NovaPOMDP),
                                        ct.c_uint,                              # numThreads
                                        ct.POINTER(ct.c_float),                 # initialGamma
                                        ct.POINTER(ct.POINTER(pav.POMDPAlphaVectors)))  # policy
_nova.pomdp_pbvi_uninitialize_gpu.argtypes = tuple([ct.POINTER(NovaPOMDP)])
_nova.pomdp_pbvi_update_gpu.argtypes = (ct.POINTER(NovaPOMDP),
                                        ct.c_uint)              # numThreads
_nova.pomdp_pbvi_get_policy_gpu.argtypes = (ct.POINTER(NovaPOMDP),
                                            ct.POINTER(ct.POINTER(pav.POMDPAlphaVectors)))  # policy


# Functions from 'pomdp_perseus_gpu.h'.


# Functions from 'pomdp_expand_gpu.h'.
_nova.pomdp_expand_random_gpu.argtypes = (ct.POINTER(NovaPOMDP),
                                            ct.c_uint,              # numThreads
                                            ct.c_uint,              # numDesiredBeliefPoints
                                            ct.POINTER(ct.c_uint),  # maxNonZeroValues
                                            ct.POINTER(ct.c_float)) # Bnew

