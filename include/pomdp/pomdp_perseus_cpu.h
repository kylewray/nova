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


#ifndef POMDP_PERSEUS_CPU_H
#define POMDP_PERSEUS_CPU_H


#include "pomdp.h"


/**
 *  Execute the entire Perseus process for the infinite horizon POMDP model specified using the CPU.
 *  @param  pomdp   The POMDP object.
 *  @param  Gamma   The resultant policy; set of alpha vectors (r-n array).
                    This will be modified.
 *  @param  pi      The resultant policy; one action for each alpha-vector (r-array).
 *                  This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_complete_cpu(POMDP *pomdp, float *Gamma, unsigned int *pi);

/**
 *  Step 1/3: The initialization step of Perseus. This sets up the Gamma and pi variables.
 *  @param  pomdp   The POMDP object.
 *  @param  Gamma   The resultant policy; set of alpha vectors (r-n array). This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_initialize_cpu(POMDP *pomdp, float *Gamma);

/**
 *  Step 2/3: Execute Perseus for the infinite horizon POMDP model specified.
 *  @param  pomdp   The POMDP object.
 *  @param  Gamma   The resultant policy; set of alpha vectors (r-n array). This will be modified.
 *  @param  pi      The resultant policy; one action for each alpha-vector (r-array). This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_execute_cpu(POMDP *pomdp, float *Gamma, unsigned int *pi);

/**
 *  Step 3/3: The uninitialization step of Perseus. This sets up the Gamma and pi variables.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_uninitialize_cpu(POMDP *pomdp);

/**
 *  The update step of Perseus. This applies the Perseus procedure once.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_update_cpu(POMDP *pomdp);

/**
 *  The get resultant policy step of Perseus. This retrieves the alpha-vectors (Gamma) and
 *  corresponding actions (pi).
 *  @param  pomdp   The POMDP object.
 *  @param  Gamma   The resultant policy; set of alpha vectors (r-n array). This will be modified.
 *  @param  pi      The resultant policy; one action for each alpha-vector (r-array).
                    This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_get_policy_cpu(POMDP *pomdp, float *Gamma, unsigned int *pi);


#endif // POMDP_PERSEUS_CPU_H

