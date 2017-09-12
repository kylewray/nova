""" The MIT License (MIT)

    Copyright (c) 2017 Kyle Hollins Wray, University of Massachusetts

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


class NovaSSPLRTDPCPU(ct.Structure):
    """ The C struct SSPLRTDPCPU object. """

    _fields_ = [("VInitial", ct.POINTER(ct.c_float)),
                ("trials", ct.c_uint),
                ("maxStackSize", ct.c_uint),
                ("currentTrial", ct.c_uint),
                ("currentHorizon", ct.c_uint),
                ("V", ct.POINTER(ct.c_float)),
                ("pi", ct.POINTER(ct.c_uint)),
                ]


_nova.ssp_lrtdp_initialize_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                          ct.POINTER(NovaSSPLRTDPCPU))

_nova.ssp_lrtdp_execute_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                       ct.POINTER(NovaSSPLRTDPCPU),
                                       ct.POINTER(mvf.MDPValueFunction))

_nova.ssp_lrtdp_uninitialize_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                            ct.POINTER(NovaSSPLRTDPCPU))

_nova.ssp_lrtdp_update_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                      ct.POINTER(NovaSSPLRTDPCPU))

_nova.ssp_lrtdp_get_policy_cpu.argtypes = (ct.POINTER(mdp.MDP),
                                          ct.POINTER(NovaSSPLRTDPCPU),
                                          ct.POINTER(mvf.MDPValueFunction))



