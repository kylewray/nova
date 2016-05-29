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


#ifndef POMDP_ALPHA_VECTORS_H
#define POMDP_ALPHA_VECTORS_H


namespace nova {

/*
 *  A structure for POMDP alpha-vector policies within nova.
 *  @param  n       The number of states in the POMDP.
 *  @param  m       The number of actions in the POMDP.
 *  @param  r       The number of alpha vectors.
 *  @param  Gamma   The values of each alpha-vector (r-n array).
 *  @param  pi      The action to take at each alpha-vector (r array).
 */
typedef struct NovaPOMDPAlphaVectors {
    unsigned int n;
    unsigned int m;
    unsigned int r;
    float *Gamma;
    unsigned int *pi;
} POMDPAlphaVectors;

/**
 *  Compute the value of a belief state.
 *  @param  policy  The POMDPAlphaVectors object.
 *  @param  b       The belief state (n array).
 *  @param  Vb      The optimal value of the belief state. This will be modified.
 *  @param  a       The optimal action of the belief state. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_alpha_vectors_value_and_action(const POMDPAlphaVectors *policy,
        const float *b, float &Vb, unsigned int &a);

/**
 *  Free the memory for *both* the policy's internal arrays *and* the policy itself.
 *  @param  policy  The resultant set of alpha-vectors. Arrays within will be freed.
 *                  This will be freed and modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_alpha_vectors_uninitialize(POMDPAlphaVectors *&policy);

};


#endif // POMDP_ALPHA_VECTORS_H


