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


class NovaPOMDPAlphaVectors(ct.Structure):
    """ The C struct POMDPAlphaVectors object. """

    _fields_ = [("n", ct.c_uint),
                ("m", ct.c_uint),
                ("r", ct.c_uint),
                ("Gamma", ct.POINTER(ct.c_float)),
                ("pi", ct.POINTER(ct.c_uint)),
                ]


# Functions from 'pomdp_alpha_vectors.h'.
_nova.pomdp_alpha_vectors_initialize.argtypes = (ct.POINTER(NovaPOMDPAlphaVectors),
                                                 ct.c_uint,      # n
                                                 ct.c_uint,      # m
                                                 ct.c_uint)      # r
_nova.pomdp_alpha_vectors_value_and_action.argtypes = (ct.POINTER(NovaPOMDPAlphaVectors),
                                        ct.POINTER(ct.c_float),                 # b
                                        ct.POINTER(ct.c_float),                 # Vb
                                        ct.POINTER(ct.c_uint))                  # a
_nova.pomdp_alpha_vectors_uninitialize.argtypes = tuple([ct.POINTER(NovaPOMDPAlphaVectors)])


