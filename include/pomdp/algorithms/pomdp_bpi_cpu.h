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


#ifndef POMDP_BPI_CPU_H
#define POMDP_BPI_CPU_H


#include "pomdp.h"

namespace nova {

/**
 *  Execute the entire BPI process for the infinite horizon POMDP model specified using the CPU.
 *  @param  pomdp   The POMDP object.
 *  @param  fsc     The resultant policy represented as a finite state controller.
 *                  This will be created and modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_bpi_complete_cpu(POMDP *pomdp, POMDPFSC *&fsc);

/**
 *  Step 1/3: The initialization step of PBVI. This sets up the fsc variables.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_bpi_initialize_cpu(POMDP *pomdp);

/**
 *  Step 2/3: Execute BPI for the infinite horizon POMDP model specified.
 *  @param  pomdp   The POMDP object.
 *  @param  fsc     The resultant policy represented as a finite state controller.
 *                  This will be created and modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_bpi_execute_cpu(POMDP *pomdp, POMDPFSC *&fsc);

/**
 *  Step 3/3: The uninitialization step of BPI. This frees the fsc variables.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_bpi_uninitialize_cpu(POMDP *pomdp);

/**
 *  The update step of BPI. This applies the update step of BPI once.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_bpi_update_cpu(POMDP *pomdp);

/**
 *  The get resultant policy step of BPI. This creates the sparse representation of
 *  the FSC to be returned to the user.
 *  @param  pomdp   The POMDP object.
 *  @param  fsc     The resultant policy represented as a finite state controller.
 *                  This will be created and modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_bpi_get_policy_cpu(POMDP *pomdp, POMDPFSC *&fsc);

};

 
#endif // POMDP_PBVI_CPU_H


