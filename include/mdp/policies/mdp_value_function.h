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


#ifndef MDP_VALUE_FUNCTION_H
#define MDP_VALUE_FUNCTION_H


/*
 *  A structure for MDP value function policies within nova.
 *  @param  n   The number of states in the MDP.
 *  @param  m   The number of actions in the MDP.
 *  @param  r   The number of relevant states in the solution. If r == 0,
 *              then all states are used.
 *  @param  S   The set of relevant states (r array). If this r == 0,
 *              then this is null, and V and pi are n arrays.
 *  @param  V   The values of the relevant states (r array or n array).
 *  @param  pi  The action to at each relevant state (r array or n array).
 */
typedef struct NovaMDPValueFunction {
    unsigned int n;
    unsigned int m;
    unsigned int r;
    unsigned int *S;
    float *V;
    unsigned int *pi;
} MDPValueFunction;

/**
 *  Free the memory for *only* the policy's internal arrays.
 *  @param  policy  The resultant value function. Arrays within will be freed.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_value_function_free(MDPValueFunction *policy);


#endif // MDP_VALUE_FUNCTION_H



