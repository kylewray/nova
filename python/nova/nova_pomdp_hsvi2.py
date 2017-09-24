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

import pomdp
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


class NovaPOMDPHSVI2CPU(ct.Structure):
    """ The C struct NovaPOMDPHSVI2CPU object. """

    _fields_ = [("trials", ct.c_uint),
                ("epsilon", ct.c_float),
                ("delta", ct.c_float),
                ("pruneGrowthThreshold", ct.c_float),
                ("maxAlphaVectors", ct.c_uint),
                ("currentTrial", ct.c_uint),
                ("lowerGammaSize", ct.c_uint),
                ("lowerGammaSizeLastPruned", ct.c_uint),
                ("lowerGamma", ct.POINTER(ct.c_float)),
                ("lowerPi", ct.POINTER(ct.c_uint)),
                ("upperGammaSize", ct.c_uint),
                ("upperGammaSizeLastPruned", ct.c_uint),
                ("upperGammaB", ct.POINTER(ct.c_float)),
                ("upperGammaHVb", ct.POINTER(ct.c_float)),
                ]


_nova.pomdp_hsvi2_initialize_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                            ct.POINTER(NovaPOMDPHSVI2CPU))

_nova.pomdp_hsvi2_execute_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                         ct.POINTER(NovaPOMDPHSVI2CPU),
                                         ct.POINTER(pav.POMDPAlphaVectors))

_nova.pomdp_hsvi2_uninitialize_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                              ct.POINTER(NovaPOMDPHSVI2CPU))

_nova.pomdp_hsvi2_update_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                        ct.POINTER(NovaPOMDPHSVI2CPU))

_nova.pomdp_hsvi2_get_policy_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                            ct.POINTER(NovaPOMDPHSVI2CPU),
                                            ct.POINTER(pav.POMDPAlphaVectors))

