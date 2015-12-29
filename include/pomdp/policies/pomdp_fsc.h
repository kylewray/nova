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


#ifndef POMDP_FSC_H
#define POMDP_FSC_H


namespace nova {

/*
 *  A structure for POMDP Finite State Controller (FSC) policies within nova.
 *
 *  @param  n       The number of nodes in the FSC.
 *  @param  m       The number of actions.
 *  @param  z       The number of possible observations made at nodes.
 *  @param  na      The maximum number of non-zero available actions.
 *  @param  ns      The maximum number of non-zero successor nodes.
 *  @param  A       The indexes of the available actions (n-na array).
 *  @param  psi     The probability of taking actions at nodes Pr(a|n) (n-na array).
 *  @param  S       The indexes of the successor nodes (n-m-z-ns array).
 *  @param  eta     The probability of node transitions Pr(n'|n,a,z)  (n-m-z-ns array).
 */
typedef struct NovaPOMDPFSC {
    unsigned int n;
    unsigned int m;
    unsigned int z;
    unsigned int na;
    unsigned int ns;
    int *A;
    float *psi;
    int *S;
    float *eta;
} POMDPFSC;

};


#endif // POMDP_FSC_H

