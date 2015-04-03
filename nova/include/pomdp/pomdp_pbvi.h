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


#ifndef NOVA_POMDP_PBVI_H
#define NOVA_POMDP_PBVI_H


/**
 * Execute PBVI for the infinite horizon POMDP model specified, except this time we
 * limit actions taken at a state to within an array of available actions. Also, adjust A
 * after completion to be within slack for each belief point.
 * @param	n					The number of states.
 * @param	m					The number of actions, in total, that are possible.
 * @param	z					The number of observations.
 * @param	r					The number of belief points.
 * @param	maxNonZeroBeliefs	The maximum number of non-zero belief states.
 * @param	maxSuccessors		The maximum number of successor states possible.
 * @param	d_B					A r-n array, consisting of r sets of n-vector belief distributions.
 *  							(Device-side pointer.)
 * @param	d_T					A mapping of state-action-state triples (n-m-n array) to a
 * 								transition probability. (Device-side pointer.)
 * @param	d_O					A mapping of action-state-observations triples (m-n-z array) to a
 * 								transition probability. (Device-side pointer.)
 * @param	d_R					A mapping of state-action triples (n-m array) to a reward.
 * 								(Device-side pointer.)
 * @param	d_available			A mapping of belief-action pairs (r-m array) to a boolean if the action
 * 								is available at that belief state or not. (Device-side pointer.)
 * @param	d_nonZeroBeliefs	A mapping from beliefs to an array of state indexes.
 * 								(Device-side pointer.)
 * @param	d_successors		A mapping from state-action pairs to successor state indexes.
 * 								(Device-side pointer.)
 * @param	gamma				The discount factor in [0.0, 1.0).
 * @param	horizon				How many time steps to iterate.
 * @param	numThreads			The number of CUDA threads per block. Use 128, 256, 512, or 1024 (multiples of 32).
 * @param	Gamma				The resultant policy; set of alpha vectors (r-n array). This will be modified.
 * @param	pi					The resultant policy; one action for each alpha-vector (r-array).
 * 								This will be modified.
 * @return	Returns 0 upon success, 1 otherwise.
 */
int nova_pomdp_pbvi(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
		const float *d_B, const float *d_T, const float *d_O, const float *d_R,
		const bool *d_available, const int *d_nonZeroBeliefs, const int *d_successors,
		float gamma, unsigned int horizon,
		unsigned int numThreads,
		float *Gamma, unsigned int *pi);

/**
 * The initialization step of PBVI. This sets up the Gamma, pi, alphaBA, and numBlocks variables.
 * @param	n				The number of states.
 * @param	m				The number of actions, in total, that are possible.
 * @param	r				The number of belief points.
 * @param	numThreads		The number of CUDA threads per block. Use 128, 256, 512, or 1024 (multiples of 32).
 * @param	Gamma			The resultant policy; set of alpha vectors (r-n array).
 * @param	d_Gamma			An r-n array of the alpha-vectors. (Device-side pointer.) This will be modified.
 * @param	d_GammaPrime	An r-n array of the alpha-vectors (copied). (Device-side pointer.) This will be modified.
 * @param	d_pi			An r-array of the actions at each belief. (Device-side pointer.) This will be modified.
 * @param	d_piPrime		An r-array of the actions at each belief (copied). (Device-side pointer.) This will be modified.
 * @param	d_alphaBA		A set of intermediate alpha-vectors. (Device-side pointer.) This will be modified.
 * @param	numBlocks		The number of blocks to execute. This will be modified.
 * @return	Returns 0 upon success, 1 otherwise.
 */
int nova_pomdp_pbvi_initialize(unsigned int n, unsigned int m, unsigned int r, unsigned int numThreads,
		float *Gamma,
		float *&d_Gamma, float *&d_GammaPrime, unsigned int *&d_pi, unsigned int *&d_piPrime, float *&d_alphaBA,
		unsigned int *numBlocks);

/**
 * The update step of PBVI. This applies the PBVI procedure once.
 * @param	n					The number of states.
 * @param	m					The number of actions, in total, that are possible.
 * @param	z					The number of observations.
 * @param	r					The number of belief points.
 * @param	maxNonZeroBeliefs	The maximum number of non-zero belief states.
 * @param	maxSuccessors		The maximum number of successor states possible.
 * @param	d_B					A r-n array, consisting of r sets of n-vector belief distributions.
 *  							(Device-side pointer.)
 * @param	d_T					A mapping of state-action-state triples (n-m-n array) to a
 * 								transition probability. (Device-side pointer.)
 * @param	d_O					A mapping of action-state-observations triples (m-n-z array) to a
 * 								transition probability. (Device-side pointer.)
 * @param	d_R					A mapping of state-action triples (n-m array) to a reward.
 * 								(Device-side pointer.)
 * @param	d_available			A mapping of belief-action pairs (r-m array) to a boolean if the action
 * 								is available at that belief state or not. (Device-side pointer.)
 * @param	d_nonZeroBeliefs	A mapping from beliefs to an array of state indexes.
 * 								(Device-side pointer.)
 * @param	d_successors		A mapping from state-action pairs to successor state indexes.
 * 								(Device-side pointer.)
 * @param	gamma				The discount factor in [0.0, 1.0).
 * @param	currentHorizon		How many applications of this method have been applied so far.
 * @param	numThreads			The number of CUDA threads per block. Use 128, 256, 512, or 1024 (multiples of 32).
 * @param	numBlocks			The number of blocks to execute.
 * @param	d_Gamma				An r-n array of the alpha-vectors. (Device-side pointer.)
 * @param	d_GammaPrime		An r-n array of the alpha-vectors (copied). (Device-side pointer.)
 * @param	d_pi				An r-array of the actions at each belief. (Device-side pointer.)
 * @param	d_piPrime			An r-array of the actions at each belief (copied). (Device-side pointer.)
 * @param	d_alphaBA			A set of intermediate alpha-vectors. (Device-side pointer.)
 * @return	Returns 0 upon success, 1 otherwise.
 */
int nova_pomdp_pbvi_update(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
		const float *d_B, const float *d_T, const float *d_O, const float *d_R,
		const bool *d_available, const int *d_nonZeroBeliefs, const int *d_successors,
		float gamma, unsigned int currentHorizon, unsigned int numThreads, unsigned int numBlocks,
		float *d_Gamma, float *d_GammaPrime, unsigned int *d_pi, unsigned int *d_piPrime, float *d_alphaBA);

/**
 * The get result step of PBVI. This retrieves the alpha-vectors (Gamma) and corresponding actions (pi).
 * @param	n				The number of states.
 * @param	r				The number of belief points.
 * @param	horizon			How many time steps to iterate.
 * @param	d_Gamma			An r-n array of the alpha-vectors. (Device-side pointer.)
 * @param	d_GammaPrime	An r-n array of the alpha-vectors (copied). (Device-side pointer.)
 * @param	d_pi			An r-array of the actions at each belief. (Device-side pointer.)
 * @param	d_piPrime		An r-array of the actions at each belief (copied). (Device-side pointer.)
 * @param	d_alphaBA		A set of intermediate alpha-vectors. (Device-side pointer.)
 * @param	Gamma			The resultant policy; set of alpha vectors (r-n array). This will be modified.
 * @param	pi				The resultant policy; one action for each alpha-vector (r-array). This will be modified.
 * @return	Returns 0 upon success, 1 otherwise.
 */
int nova_pomdp_pbvi_get_result(unsigned int n, unsigned int r, unsigned int horizon,
		const float *d_Gamma, const float *d_GammaPrime, const unsigned int *d_pi, const unsigned int *d_piPrime,
		float *Gamma, unsigned int *pi);

/**
 * The uninitialization step of PBVI. This sets up the Gamma, pi, alphaBA, and numBlocks variables.
 * @param	d_Gamma			An r-n array of the alpha-vectors. (Device-side pointer.) This will be modified.
 * @param	d_GammaPrime	An r-n array of the alpha-vectors (copied). (Device-side pointer.) This will be modified.
 * @param	d_pi			An r-array of the actions at each belief. (Device-side pointer.) This will be modified.
 * @param	d_piPrime		An r-array of the actions at each belief (copied). (Device-side pointer.) This will be modified.
 * @param	d_alphaBA		A set of intermediate alpha-vectors. (Device-side pointer.) This will be modified.
 * @return	Returns 0 upon success, 1 otherwise.
 */
int nova_pomdp_pbvi_uninitialize(float *&d_Gamma, float *&d_GammaPrime,
		unsigned int *&d_pi, unsigned int *&d_piPrime, float *&d_alphaBA);

/**
 * Initialize CUDA available actions object.
 * @param	m			The number of actions, in total, that are possible.
 * @param	r			The number of belief points.
 * @param	available	A r-m array, consisting of r sets of m-vector action availability booleans.
 * @param	d_available	A r-m array, consisting of r sets of m-vector action availability booleans.
 *  					(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_initialize_available(unsigned int m, unsigned int r, const bool *available, bool *&d_available);

/**
 * Uninitialize CUDA available actions object.
 * @param	d_available	A r-m array, consisting of r sets of m-vector action availability booleans.
 *  					(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_uninitialize_available(float *&d_available);

/**
 * Initialize CUDA belief points object.
 * @param	n			The number of states.
 * @param	r			The number of belief points.
 * @param	B			A r-n array, consisting of r sets of n-vector belief distributions.
 * @param	d_B			A r-n array, consisting of r sets of n-vector belief distributions.
 *  					(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_initialize_belief_points(unsigned int n, unsigned int r, const float *B, float *&d_B);

/**
 * Uninitialize CUDA belief points object.
 * @param	d_B			A r-n array, consisting of r sets of n-vector belief distributions.
 *  					(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_uninitialize_belief_points(float *&d_B);

/**
 * Initialize CUDA state transitions object.
 * @param	n			The number of states.
 * @param	m			The number of actions, in total, that are possible.
 * @param	T			A mapping of state-action-state triples (n-m-n array) to a
 * 						transition probability.
 * @param	d_T			A mapping of state-action-state triples (n-m-n array) to a
 * 						transition probability. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_initialize_state_transitions(unsigned int n, unsigned int m, const float *T, float *&d_T);

/**
 * Uninitialize CUDA state transitions object.
 * @param	d_T			A mapping of state-action-state triples (n-m-n array) to a
 * 						transition probability. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_uninitialize_state_transitions(float *&d_T);

/**
 * Initialize CUDA observation transitions object.
 * @param	n			The number of states.
 * @param	m			The number of actions, in total, that are possible.
 * @param	z			The number of observations.
 * @param	O			A mapping of action-state-observation triples (m-n-z array) to a
 * 						transition probability.
 * @param	d_O			A mapping of action-state-observation triples (m-n-z array) to a
 * 						transition probability. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_initialize_observation_transitions(unsigned int n, unsigned int m, unsigned int z, const float *O, float *&d_O);

/**
 * Uninitialize CUDA observation transitions object.
 * @param	d_O			A mapping of action-state-observation triples (m-n-z array) to a
 * 						transition probability. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_uninitialize_observation_transitions(float *&d_O);

/**
 * Initialize CUDA rewards object.
 * @param	n			The number of states.
 * @param	m			The number of actions, in total, that are possible.
 * @param	R			A mapping of state-action pairs (n-m array) to a reward.
 * @param	d_R			A mapping of state-action pairs (n-m array) to a reward.
 * 						(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_initialize_rewards(unsigned int n, unsigned int m, const float *R, float *&d_R);

/**
 * Uninitialize CUDA rewards object.
 * @param	d_R			A mapping of state-action pairs (n-m array) to a reward.
 * 						(Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_uninitialize_rewards(float *&d_R);

/**
 * Initialize CUDA non-zero belief states object.
 * @param	r					The number of belief points.
 * @param	maxNonZeroBeliefs	The maximum number of non-zero belief states.
 * @param	nonZeroBeliefs		A mapping of beliefs to an array of state indexes;
 * 								-1 denotes the end of the array.
 * @param	d_nonZeroBeliefs	A mapping of beliefs to an array of state indexes;
 * 								-1 denotes the end of the array. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_initialize_nonzero_beliefs(unsigned int r, unsigned int maxNonZeroBeliefs,
		int *nonZeroBeliefs, int *&d_nonZeroBeliefs);

/**
 * Uninitialize CUDA non-zero belief states object.
 * @param	d_NonZeroBeliefs	A mapping of beliefs to an array of state indexes;
 * 								-1 denotes the end of the array. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_uninitialize_nonzero_beliefs(int *&d_nonZeroBeliefs);

/**
 * Initialize CUDA by transferring all of the constant LPOMDP model information to the device.
 * @param	n				The number of states.
 * @param	m				The number of actions, in total, that are possible.
 * @param	maxSuccessors	The maximum number of successor states.
 * @param	successors		A mapping of state-action pairs a set of possible successor states;
 * 							-1 denotes the end of the array.
 * @param	d_successors	A mapping of state-action pairs a set of possible successor states;
 * 							-1 denotes the end of the array. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_initialize_successors(unsigned int n, unsigned int m, unsigned int maxSuccessors,
		int *successors, int *&d_successors);

/**
 * Uninitialize CUDA by transferring all of the constant LPOMDP model information to the device.
 * @param	d_successors	A mapping of state-action pairs a set of possible successor states;
 * 							-1 denotes the end of the array. (Device-side pointer.)
 * @return	Returns 0 upon success; -1 if invalid arguments were passed; -3 if an error with
 * 			the CUDA functions arose.
 */
int nova_pomdp_pbvi_uninitialize_successors(int *&d_successors);


#endif // NOVA_POMDP_PBVI_H
