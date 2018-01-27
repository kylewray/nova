/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2017 Kyle Hollins Wray, University of Massachusetts
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


#ifndef NOVA_POMDP_CBNLP_H
#define NOVA_POMDP_CBNLP_H


#include <nova/pomdp/pomdp.h>
#include <nova/pomdp/policies/pomdp_stochastic_fsc.h>
#include <nova/constants.h>

namespace nova {

/**
 *  The necessary variables to perform belief-infused non-linear programming (NLP) on a POMDP within nova.
 *
 *  Note: Total number of nodes is (k + r) in the 'policy' variable. The first k are FSC nodes,
 *  the remaining r are the belief points following the order of B.
 *
 *  @param  path    The path where the AMPL files will be saved.
 *  @param  command The command to execute the AMPL solver; the policy will be extracted from stdout.
 *  @param  k       The number of controller nodes desired in the final policy.
 *  @param  r       The number of belief points to find and add during belief infusion.
 *  @param  B       The set of r belief points each over the n states.
 *  @param  lambda  The probabilistic weight placed on doing PBVI. FSC is (1-lambda).
 *  @param  policy  The resultant (intermediate) controller action and transition probabilities.
 *  @param  V       The resultant values for each controller node and state pair.
 */
typedef struct NovaPOMDPCBNLP {
    char *path;
    char *command;
    unsigned int k;
    unsigned int r;
    float *B;
    float lambda;
    float *policy;
    float *V;
} POMDPCBNLP;

/**
 *  Execute all the belief-infused NLP steps for the infinite horizon POMDP model specified.
 *  @param  pomdp   The POMDP object.
 *  @param  cbnlp   The POMDPCBNLP object containing algorithm variables.
 *  @param  policy  The resultant controller node probabilities. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_cbnlp_execute(const POMDP *pomdp, POMDPCBNLP *bi, POMDPStochasticFSC *policy);

/**
 *  Step 1/3: The initialization step of the belief-infused NLP solution. This saves the POMDP as an AMLP data file.
 *  @param  pomdp   The POMDP object.
 *  @param  cbnlp   The POMDPCBNLP object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_cbnlp_initialize(const POMDP *pomdp, POMDPCBNLP *bi);

/**
 *  Step 2/3: Perform the solving of the belief-infused NLP. There is only one call to update for the NLP solution.
 *  @param  pomdp   The POMDP object.
 *  @param  cbnlp   The POMDPCBNLP object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_cbnlp_update(const POMDP *pomdp, POMDPCBNLP *bi);

/**
 *  Step 3/4: The get resultant policy step of the belief-infused NLP solution. This parses the solver result.
 *  @param  pomdp   The POMDP object.
 *  @param  cbnlp   The POMDPCBNLP object containing algorithm variables.
 *  @param  policy  The resultant controller node probabilities. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_cbnlp_get_policy(const POMDP *pomdp, POMDPCBNLP *pbvi, POMDPStochasticFSC *policy);

/**
 *  Step 4/4: The uninitialization step of PBVI. This frees variables and removes temporary files.
 *  @param  pomdp   The POMDP object.
 *  @param  cbnlp   The POMDPCBNLP object containing algorithm variables.
 *  @param  policy  The resultant controller node probabilities. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_cbnlp_uninitialize(const POMDP *pomdp, POMDPCBNLP *bi);

};

 
#endif // NOVA_POMDP_CBNLP_H



