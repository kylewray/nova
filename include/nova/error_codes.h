/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2015 Kyle Hollins Wray, University of Massachusetts
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *  the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 *  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 *  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 *  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#ifndef NOVA_ERROR_CODES_H
#define NOVA_ERROR_CODES_H


namespace nova {

// Successes.
#define NOVA_SUCCESS                                0

// Critical errors due to POMDP model, the CPU, or the GPU.
#define NOVA_ERROR_INVALID_DATA                     1
#define NOVA_ERROR_INVALID_CUDA_PARAM               2
#define NOVA_ERROR_DEVICE_MALLOC                    3
#define NOVA_ERROR_MEMCPY_TO_DEVICE                 4
#define NOVA_ERROR_MEMCPY_TO_HOST                   5
#define NOVA_ERROR_DEVICE_FREE                      6
#define NOVA_ERROR_KERNEL_EXECUTION                 7
#define NOVA_ERROR_DEVICE_SYNCHRONIZE               8

// Other results, warnings, or errors which are possible during run time.
#define NOVA_CONVERGED                              9
#define NOVA_WARNING_INVALID_BELIEF                 10
#define NOVA_ERROR_OUT_OF_MEMORY                    11
#define NOVA_ERROR_POLICY_CREATION                  12
#define NOVA_WARNING_APPROXIMATE_SOLUTION           13
#define NOVA_ERROR_EMPTY_CONTAINER                  14
#define NOVA_ERROR_FAILED_TO_OPEN_FILE              15
#define NOVA_ERROR_EXECUTING_COMMAND                16

};


#endif // NOVA_ERROR_CODES_H

