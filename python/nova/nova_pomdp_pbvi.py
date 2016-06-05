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


class NovaPOMDPPBVICPU(ct.Structure):
    """ The C struct NovaPOMDPPBVICPU object. """

    _fields_ = [("GammaInitial", ct.POINTER(ct.c_float)),
                ("currentHorizon", ct.c_uint),
                ("Gamma", ct.POINTER(ct.c_float)),
                ("GammaPrime", ct.POINTER(ct.c_float)),
                ("pi", ct.POINTER(ct.c_uint)),
                ]


_nova.pomdp_pbvi_initialize_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                            ct.POINTER(NovaPOMDPPBVICPU))

_nova.pomdp_pbvi_execute_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                         ct.POINTER(NovaPOMDPPBVICPU),
                                         ct.POINTER(pav.POMDPAlphaVectors))

_nova.pomdp_pbvi_uninitialize_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                              ct.POINTER(NovaPOMDPPBVICPU))

_nova.pomdp_pbvi_update_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                        ct.POINTER(NovaPOMDPPBVICPU))

_nova.pomdp_pbvi_get_policy_cpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                            ct.POINTER(NovaPOMDPPBVICPU),
                                            ct.POINTER(pav.POMDPAlphaVectors))


class NovaPOMDPPBVIGPU(ct.Structure):
    """ The C struct NovaPOMDPPBVIGPU object. """

    _fields_ = [("GammaInitial", ct.POINTER(ct.c_float)),
                ("numThreads", ct.c_uint),
                ("currentHorizon", ct.c_uint),
                ("d_Gamma", ct.POINTER(ct.c_float)),
                ("d_GammaPrime", ct.POINTER(ct.c_float)),
                ("d_pi", ct.POINTER(ct.c_uint)),
                ("d_alphaBA", ct.POINTER(ct.c_float)),
                ]


_nova.pomdp_pbvi_initialize_gpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                            ct.POINTER(NovaPOMDPPBVIGPU))

_nova.pomdp_pbvi_execute_gpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                         ct.POINTER(NovaPOMDPPBVIGPU),
                                         ct.POINTER(pav.POMDPAlphaVectors))

_nova.pomdp_pbvi_uninitialize_gpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                              ct.POINTER(NovaPOMDPPBVIGPU))

_nova.pomdp_pbvi_update_gpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                        ct.POINTER(NovaPOMDPPBVIGPU))

_nova.pomdp_pbvi_get_policy_gpu.argtypes = (ct.POINTER(pomdp.POMDP),
                                            ct.POINTER(NovaPOMDPPBVIGPU),
                                            ct.POINTER(pav.POMDPAlphaVectors))

