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

#include "../../include/pomdp/pomdp_pbvi.h"

#include <stdio.h>

// This is not C++0x, unfortunately.
#define nullptr NULL

// This is determined by hardware, so what is below is a 'safe' guess. If this is
// off, the program might return 'nan' or 'inf'. These come from IEEE floating-point
// standards.
#define FLT_MAX 1e+35
#define FLT_MIN -1e+35
#define FLT_ERR_TOL 1e-9

__global__ void nova_pomdp_pbvi_initialize_alphaBA(unsigned int n, unsigned int m, unsigned int r, const float *R, float *alphaBA)
{
	unsigned int beliefIndex = blockIdx.x;
	unsigned int action = blockIdx.y;

	if (beliefIndex >= r || action >= m) {
		return;
	}

	// Compute Gamma_{a,*} and set it to the first value of alphaBA. Stride here.
	for (unsigned int s = threadIdx.x; s < n; s += blockDim.x) {
		alphaBA[beliefIndex * m * n + action * n + s] = R[s * m + action];
	}
}

__global__ void nova_pomdp_pbvi_compute_alphaBA(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
		const float *B, const float *T, const float *O, const float *R,
		const bool *available, const int *nonZeroBeliefs, const int *successors,
		float gamma,
		const float *Gamma, const unsigned int *pi,
		float *alphaBA)
{
	// Since float and unsigned int are 4 bytes each, and we need each array to be the size of
	// the number of threads, we will need to call this with:
	// sizeof(float) * numThreads + sizeof(unsigned int * numThreads.
	// Note: blockDim.x == numThreads
	extern __shared__ float sdata[];
	float *maxAlphaDotBeta = (float *)sdata;
	unsigned int *maxAlphaIndex = (unsigned int *)&maxAlphaDotBeta[blockDim.x];

	maxAlphaDotBeta[threadIdx.x] = FLT_MIN;
	maxAlphaIndex[threadIdx.x] = 0;

	__syncthreads();

	unsigned int beliefIndex = blockIdx.x;
	unsigned int action = blockIdx.y;
	unsigned int observation = blockIdx.z;

	if (beliefIndex >= r || action >= m || observation >= z) {
		return;
	}

	// Compute the max alpha vector from Gamma, given the fixed action and observation.
	// Note: this is the max w.r.t. just the strided elements. The reduction will
	// be computed afterwards for the max over all alpha-vectors.

	for (unsigned int alphaIndex = threadIdx.x; alphaIndex < r; alphaIndex += blockDim.x) {
		float alphaDotBeta = 0.0f;

		for (unsigned int i = 0; i < maxNonZeroBeliefs; i++) {
			int s = nonZeroBeliefs[beliefIndex * maxNonZeroBeliefs + i];
			if (s < 0) {
				break;
			}

			// We compute the value of this state in the alpha-vector, then multiply it by the belief, and add it to
			// the current dot product value for this alpha-vector.
			float value = 0.0f;
			for (unsigned int j = 0; j < maxSuccessors; j++) {
				int sp = successors[s * m * maxSuccessors + action * maxSuccessors + j];
				if (sp < 0) {
					break;
				}
				value += T[s * m * n + action * n + sp] * O[action * n * z + sp * z + observation] * Gamma[alphaIndex * n + sp];
			}
			value *= gamma;

			__syncthreads();

			alphaDotBeta += value * B[beliefIndex * n + s];
		}

		__syncthreads();

		// Store the maximal value and index.
		if (alphaIndex == threadIdx.x || alphaDotBeta > maxAlphaDotBeta[threadIdx.x]) {
			maxAlphaDotBeta[threadIdx.x] = alphaDotBeta; //[action * z * r + observation * r + alphaIndex];
			maxAlphaIndex[threadIdx.x] = alphaIndex;
		}
	}

	// Note: The above code essentially does the first add during load. It takes care of *all* the other elements
	// *outside* the number of threads we have. In other words, starting here, we already have computed part of the
	// maxAlphaDotBeta and maxAlphaIndex; we just need to finish the rest quickly, using a reduction.
	__syncthreads();

	// Use reduction to compute the max overall alpha-vector.
	for (unsigned int alphaIndex = blockDim.x / 2; alphaIndex > 0; alphaIndex >>= 1) {
		if (threadIdx.x < alphaIndex && threadIdx.x < r && threadIdx.x + alphaIndex < r) {
			if (maxAlphaDotBeta[threadIdx.x] < maxAlphaDotBeta[threadIdx.x + alphaIndex]) {
				maxAlphaDotBeta[threadIdx.x] = maxAlphaDotBeta[threadIdx.x + alphaIndex];
				maxAlphaIndex[threadIdx.x] = maxAlphaIndex[threadIdx.x + alphaIndex];
			}
		}

		__syncthreads();
	}

	// Now we can compute the alpha-vector component for this observation, since we have the max.
	// We will need to compute the dot product anyway, so let's just distribute the belief over the
	// sum over observations, and add it all up here.
	// Note: This re-uses the thread to stride over states now.
	for (unsigned int s = threadIdx.x; s < n; s += blockDim.x) {
		// We compute the value of this state in the alpha-vector, then multiply it by the belief, and add it to
		// the current dot product value for this alpha-vector.
		float value = 0.0f;
		for (unsigned int i = 0; i < maxSuccessors; i++) {
			int sp = successors[s * m * maxSuccessors + action * maxSuccessors + i];
			if (sp < 0) {
				break;
			}
			// Note: maxAlphaIndex[0] holds the maximal index value computed from the reduction above.
			value += T[s * m * n + action * n + sp] * O[action * n * z + sp * z + observation] * Gamma[maxAlphaIndex[0] * n + sp];
		}

		__syncthreads();

		alphaBA[beliefIndex * m * n + action * n + s] += gamma * value;
	}
}

__global__ void nova_pomdp_pbvi_update_step(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
		const float *B, const float *T, const float *O, const float *R,
		const bool *available, const int *nonZeroBeliefs, const int *successors,
		float gamma,
		const float *Gamma, const unsigned int *pi,
		float *alphaBA,
		float *GammaPrime, unsigned int *piPrime)
{
	// Each block will run a different belief. Our overall goal: Compute the value
	// of GammaPrime[beliefIndex * n + ???] and piPrime[beliefIndex].
	unsigned int beliefIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (beliefIndex >= r) {
		return;
	}

	// We want to find the action that maximizes the value, store it in piPrime, as well as its alpha-vector GammaPrime.
	float maxActionValue = FLT_MIN;

	for (unsigned int action = 0; action < m; action++) {
		// Only execute if the action is available.
		if (available[beliefIndex * m + action]) {
			// The potential alpha-vector has been computed, so compute the value with respect to the belief state.
			float actionValue = 0.0f;
			for (unsigned int i = 0; i < maxNonZeroBeliefs; i++) {
				int s = nonZeroBeliefs[beliefIndex * maxNonZeroBeliefs + i];
				if (s < 0) {
					break;
				}
				actionValue += alphaBA[beliefIndex * m * n + action * n + s] * B[beliefIndex * n + s];
			}

			// If this was larger, then overwrite piPrime and GammaPrime's values.
			if (actionValue > maxActionValue) {
				maxActionValue = actionValue;

				piPrime[beliefIndex] = action;
			}
		}

		__syncthreads();
	}

	for (unsigned int s = 0; s < n; s++) {
		GammaPrime[beliefIndex * n + s] = alphaBA[beliefIndex * m * n + piPrime[beliefIndex] * n + s];
	}
}

int nova_pomdp_pbvi_initialize(unsigned int n, unsigned int m, unsigned int r, unsigned int numThreads,
		float *Gamma,
		float *&d_Gamma, float *&d_GammaPrime, unsigned int *&d_pi, unsigned int *&d_piPrime, float *&d_alphaBA,
		unsigned int *numBlocks)
{
	*numBlocks = (unsigned int)((float)r / (float)numThreads) + 1;

	// Create the device-side Gamma.
	if (cudaMalloc(&d_Gamma, r * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize]: %s",
				"Failed to allocate device-side memory for Gamma.");
		return 1;
	}
	if (cudaMemcpy(d_Gamma, Gamma, r * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize]: %s",
				"Failed to copy memory from host to device for Gamma.");
		return 2;
	}

	if (cudaMalloc(&d_GammaPrime, r * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize]: %s",
				"Failed to allocate device-side memory for Gamma (prime).");
		return 3;
	}
	if (cudaMemcpy(d_GammaPrime, Gamma, r * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize]: %s",
				"Failed to copy memory from host to device for Gamma (prime).");
		return 4;
	}

	// Create the device-side pi.
	if (cudaMalloc(&d_pi, r * sizeof(unsigned int)) != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize]: %s",
				"Failed to allocate device-side memory for pi.");
		return 5;
	}
	if (cudaMalloc(&d_piPrime, r * sizeof(unsigned int)) != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize]: %s",
				"Failed to allocate device-side memory for pi (prime).");
		return 6;
	}

	// Create the device-side memory for the intermediate variable alphaBA.
	if (cudaMalloc(&d_alphaBA, r * m * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize]: %s",
				"Failed to allocate device-side memory for alphaBA.");
		return 7;
	}

	return 0;
}

int nova_pomdp_pbvi_update(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
		const float *d_B, const float *d_T, const float *d_O, const float *d_R,
		const bool *d_available, const int *d_nonZeroBeliefs, const int *d_successors,
		float gamma, unsigned int currentHorizon, unsigned int numThreads, unsigned int numBlocks,
		float *d_Gamma, float *d_GammaPrime, unsigned int *d_pi, unsigned int *d_piPrime, float *d_alphaBA)
{
	nova_pomdp_pbvi_initialize_alphaBA<<< dim3(r, m, 1), numThreads >>>(n, m, r, d_R, d_alphaBA);

	// Check if there was an error executing the kernel.
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_update]: %s",
						"Failed to execute the 'initialization of alphaBA' kernel.");
		return 1;
	}

	// Wait for the kernel to finish before looping more.
	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_update]: %s",
						"Failed to synchronize the device after 'initialization of alphaBA' kernel.");
		return 2;
	}

	nova_pomdp_pbvi_compute_alphaBA<<< dim3(r, m, z), numThreads,
				numThreads * sizeof(float) + numThreads * sizeof(unsigned int) >>>(
				n, m, z, r,
				maxNonZeroBeliefs, maxSuccessors,
				d_B, d_T, d_O, d_R,
				d_available, d_nonZeroBeliefs, d_successors,
				gamma,
				d_Gamma, d_pi,
				d_alphaBA);

	// Check if there was an error executing the kernel.
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_update]: %s",
						"Failed to execute the 'compute_alphaBA' kernel.");
		return 3;
	}

	// Wait for the kernel to finish before looping more.
	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_update]: %s",
						"Failed to synchronize the device after 'compute_alphaBA' kernel.");
		return 4;
	}

	// Execute a kernel for the first three stages of for-loops: B, A, Z, as a 3d-block,
	// and the 4th stage for-loop over Gamma as the threads.
	if (currentHorizon % 2 == 0) {
		nova_pomdp_pbvi_update_step<<< numBlocks, numThreads >>>(n, m, z, r,
				maxNonZeroBeliefs, maxSuccessors,
				d_B, d_T, d_O, d_R,
				d_available, d_nonZeroBeliefs, d_successors,
				gamma,
				d_Gamma, d_pi,
				d_alphaBA,
				d_GammaPrime, d_piPrime);
	} else {
		nova_pomdp_pbvi_update_step<<< numBlocks, numThreads >>>(n, m, z, r,
				maxNonZeroBeliefs, maxSuccessors,
				d_B, d_T, d_O, d_R,
				d_available, d_nonZeroBeliefs, d_successors,
				gamma,
				d_GammaPrime, d_piPrime,
				d_alphaBA,
				d_Gamma, d_pi);
	}

	// Check if there was an error executing the kernel.
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_update]: %s",
						"Failed to execute the 'update_step' kernel.");
		return 5;
	}

	// Wait for the kernel to finish before looping more.
	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_update]: %s",
						"Failed to synchronize the device after 'update_step' kernel.");
		return 6;
	}

	return 0;
}

int nova_pomdp_pbvi_get_result(unsigned int n, unsigned int r, unsigned int horizon,
		const float *d_Gamma, const float *d_GammaPrime, const unsigned int *d_pi, const unsigned int *d_piPrime,
		float *Gamma, unsigned int *pi)
{
	// Copy the final result of Gamma and pi to the variables. This assumes
	// that the memory has been allocated.
	if (horizon % 2 == 1) {
		if (cudaMemcpy(Gamma, d_Gamma, r * n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_get_result]: %s",
					"Failed to copy memory from device to host for Gamma.");
			return 1;
		}
		if (cudaMemcpy(pi, d_pi, r * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_get_result]: %s",
					"Failed to copy memory from device to host for pi.");
			return 2;
		}
	} else {
		if (cudaMemcpy(Gamma, d_GammaPrime, r * n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_get_result]: %s",
					"Failed to copy memory from device to host for Gamma (prime).");
			return 3;
		}
		if (cudaMemcpy(pi, d_piPrime, r * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_get_result]: %s",
					"Failed to copy memory from device to host for pi (prime).");
			return 4;
		}
	}

	return 0;
}

int nova_pomdp_pbvi_uninitialize(float *&d_Gamma, float *&d_GammaPrime,
		unsigned int *&d_pi, unsigned int *&d_piPrime, float *&d_alphaBA)
{
	if (d_Gamma != nullptr) {
		if (cudaFree(d_Gamma) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize]: %s",
					"Failed to allocate device-side memory for the Gamma (the alpha-vectors).");
		}
	}
	d_Gamma = nullptr;

	if (d_GammaPrime != nullptr) {
		if (cudaFree(d_GammaPrime) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize]: %s",
					"Failed to allocate device-side memory for the GammaPrime (the alpha-vectors' copy).");
		}
	}
	d_GammaPrime = nullptr;

	if (d_pi != nullptr) {
		if (cudaFree(d_pi) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize]: %s",
					"Failed to allocate device-side memory for the pi (the policy).");
		}
	}
	d_pi = nullptr;

	if (d_piPrime != nullptr) {
		if (cudaFree(d_piPrime) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize]: %s",
					"Failed to allocate device-side memory for the piPrime (the policy copy).");
		}
	}
	d_piPrime = nullptr;

	if (d_alphaBA != nullptr) {
		if (cudaFree(d_alphaBA) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize]: %s",
					"Failed to allocate device-side memory for alphaBA (collection of alpha-vectors).");
		}
	}
	d_alphaBA = nullptr;

	return 0;
}

int nova_pomdp_pbvi(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
		unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
		const float *d_B, const float *d_T, const float *d_O, const float *d_R,
		const bool *d_available, const int *d_nonZeroBeliefs, const int *d_successors,
		float gamma, unsigned int horizon,
		unsigned int numThreads,
		float *Gamma, unsigned int *pi)
{
	// The device pointers for the alpha-vectors: Gamma and GammaPrime.
	float *d_Gamma;
	float *d_GammaPrime;

	// The device pointers for the actions taken on each alpha-vector: pi and piPrime.
	unsigned int *d_pi;
	unsigned int *d_piPrime;

	// The device pointer for the intermediate alpha-vectors computed in the inner for loop.
	float *d_alphaBA;

	// The number of blocks to execute in the main PBVI step.
	unsigned int numBlocks = 0;

	// Ensure the data is valid.
	if (n == 0 || m == 0 || z == 0 || r == 0 || maxNonZeroBeliefs == 0 || maxSuccessors == 0 ||
			d_B == nullptr || d_T == nullptr || d_O == nullptr || d_R == nullptr ||
			d_available == nullptr || d_nonZeroBeliefs == nullptr || d_successors == nullptr ||
			gamma < 0.0 || gamma >= 1.0 || horizon < 1) {
		fprintf(stderr, "Error[nova_pomdp_pbvi]: %s", "Invalid arguments.");
		return 1;
	}

	// Ensure threads are correct.
	if (numThreads % 32 != 0) {
		fprintf(stderr, "Error[nova_pomdp_pbvi]: %s", "Invalid number of threads.");
		return 2;
	}

	nova_pomdp_pbvi_initialize(n, m, r, numThreads, Gamma, d_Gamma, d_GammaPrime, d_pi, d_piPrime, d_alphaBA, &numBlocks);

	// For each of the updates, run PBVI.
	for (int t = 0; t < horizon; t++) {
		nova_pomdp_pbvi_update(n, m, z, r, maxNonZeroBeliefs, maxSuccessors,
				d_B, d_T, d_O, d_R,
				d_available, d_nonZeroBeliefs, d_successors,
				gamma, t, numThreads, numBlocks,
				d_Gamma, d_GammaPrime, d_pi, d_piPrime, d_alphaBA);

//		fprintf(stdout, "Iteration %i of %i\n", t+1, horizon);
	}

	nova_pomdp_pbvi_get_result(n, r, horizon, d_Gamma, d_GammaPrime, d_pi, d_piPrime, Gamma, pi);

	nova_pomdp_pbvi_uninitialize(d_Gamma, d_GammaPrime, d_pi, d_piPrime, d_alphaBA);

	return 0;
}

int nova_pomdp_pbvi_initialize_available(unsigned int m, unsigned int r, const bool *available, bool *&d_available)
{
	// Ensure the data is valid.
	if (m == 0 || r == 0 || available == nullptr) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize_available]: %s", "Invalid input.");
		return 1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_available, r * m * sizeof(bool)) != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize_available]: %s",
				"Failed to allocate device-side memory for the available actions.");
		return 2;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_available, available, r * m * sizeof(bool), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[nova_pomdp_pbvi_initialize_available]: %s",
				"Failed to copy memory from host to device for the available actions.");
		return 3;
	}

	return 0;
}

int nova_pomdp_pbvi_uninitialize_available(float *&d_available)
{
	if (d_available != nullptr) {
		if (cudaFree(d_available) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize_available]: %s",
					"Failed to allocate device-side memory for the available actions.");
			return 1;
		}
	}
	d_available = nullptr;

	return 0;
}

int lpbvi_initialize_belief_points(unsigned int n, unsigned int r, const float *B, float *&d_B)
{
	// Ensure the data is valid.
	if (n == 0 || r == 0 || B == nullptr) {
		fprintf(stderr, "Error[lpbvi_initialize_belief_points]: %s", "Invalid input.");
		return 1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_B, r * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_belief_points]: %s",
				"Failed to allocate device-side memory for the belief points.");
		return 2;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_B, B, r * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_belief_points]: %s",
				"Failed to copy memory from host to device for the belief points.");
		return 3;
	}

	return 0;
}

int nova_pomdp_pbvi_uninitialize_belief_points(float *&d_B)
{
	if (d_B != nullptr) {
		if (cudaFree(d_B) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize_belief_points]: %s",
					"Failed to allocate device-side memory for the belief points.");
			return 1;
		}
	}
	d_B = nullptr;

	return 0;
}

int lpbvi_initialize_state_transitions(unsigned int n, unsigned int m, const float *T, float *&d_T)
{
	// Ensure the data is valid.
	if (n == 0 || m == 0 || T == nullptr) {
		fprintf(stderr, "Error[lpbvi_initialize_state_transitions]: %s", "Invalid input.");
		return 1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_T, n * m * n * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_state_transitions]: %s",
				"Failed to allocate device-side memory for the state transitions.");
		return 2;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_T, T, n * m * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_state_transitions]: %s",
				"Failed to copy memory from host to device for the state transitions.");
		return 3;
	}

	return 0;
}

int nova_pomdp_pbvi_uninitialize_state_transitions(float *&d_T)
{
	if (d_T != nullptr) {
		if (cudaFree(d_T) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize_state_transitions]: %s",
					"Failed to allocate device-side memory for the state transitions.");
			return 1;
		}
	}
	d_T = nullptr;

	return 0;
}

int lpbvi_initialize_observation_transitions(unsigned int n, unsigned int m, unsigned int z,
		const float *O, float *&d_O)
{
	// Ensure the data is valid.
	if (n == 0 || m == 0 || z == 0 || O == nullptr) {
		fprintf(stderr, "Error[lpbvi_initialize_observation_transitions]: %s", "Invalid input.");
		return 1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_O, m * n * z * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_observation_transitions]: %s",
				"Failed to allocate device-side memory for the observation transitions.");
		return 2;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_O, O, m * n * z * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_observation_transitions]: %s",
				"Failed to copy memory from host to device for the observation transitions.");
		return 3;
	}

	return 0;
}

int nova_pomdp_pbvi_uninitialize_observation_transitions(float *&d_O)
{
	if (d_O != nullptr) {
		if (cudaFree(d_O) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize_observation_transitions]: %s",
					"Failed to allocate device-side memory for the observation transitions.");
			return 1;
		}
	}
	d_O = nullptr;

	return 0;
}

int lpbvi_initialize_rewards(unsigned int n, unsigned int m, const float *R, float *&d_R)
{
	// Ensure the data is valid.
	if (n == 0 || m == 0 || R == nullptr) {
		fprintf(stderr, "Error[lpbvi_initialize_rewards]: %s", "Invalid input.");
		return 1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_R, n * m * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_rewards]: %s",
				"Failed to allocate device-side memory for the rewards.");
		return 2;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_R, R, n * m * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_rewards]: %s",
				"Failed to copy memory from host to device for the rewards.");
		return 3;
	}

	return 0;
}

int nova_pomdp_pbvi_uninitialize_rewards(float *&d_R)
{
	if (d_R != nullptr) {
		if (cudaFree(d_R) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize_rewards]: %s",
					"Failed to allocate device-side memory for the rewards.");
			return 1;
		}
	}
	d_R = nullptr;

	return 0;
}

int lpbvi_initialize_nonzero_beliefs(unsigned int r, unsigned int maxNonZeroBeliefs,
		int *nonZeroBeliefs, int *&d_nonZeroBeliefs)
{
	// Ensure the data is valid.
	if (r == 0 || maxNonZeroBeliefs == 0 || nonZeroBeliefs == nullptr) {
		fprintf(stderr, "Error[lpbvi_initialize_nonzero_beliefs]: %s", "Invalid input.");
		return 1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_nonZeroBeliefs, r * maxNonZeroBeliefs * sizeof(int)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_nonzero_beliefs]: %s",
				"Failed to allocate device-side memory for the non-zero belief states.");
		return 2;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_nonZeroBeliefs, nonZeroBeliefs, r * maxNonZeroBeliefs * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_nonzero_beliefs]: %s",
				"Failed to copy memory from host to device for the non-zero belief states.");
		return 3;
	}

	return 0;
}

int nova_pomdp_pbvi_uninitialize_nonzero_beliefs(int *&d_nonZeroBeliefs)
{
	if (d_nonZeroBeliefs != nullptr) {
		if (cudaFree(d_nonZeroBeliefs) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize_nonzero_beliefs]: %s",
					"Failed to allocate device-side memory for the non-zero belief states.");
			return 1;
		}
	}
	d_nonZeroBeliefs = nullptr;

	return 0;
}

int lpbvi_initialize_successors(unsigned int n, unsigned int m, unsigned int maxSuccessors,
		int *successors, int *&d_successors)
{
	// Ensure the data is valid.
	if (n == 0 || m == 0 || maxSuccessors <= 0 || successors == nullptr) {
		fprintf(stderr, "Error[lpbvi_initialize_successors]: %s", "Invalid input.");
		return 1;
	}

	// Allocate the memory on the device.
	if (cudaMalloc(&d_successors, n * m * maxSuccessors * sizeof(int)) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_successors]: %s",
				"Failed to allocate device-side memory for the successor states.");
		return 2;
	}

	// Copy the data from the host to the device.
	if (cudaMemcpy(d_successors, successors, n * m * maxSuccessors * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Error[lpbvi_initialize_successors]: %s",
				"Failed to copy memory from host to device for the successor states.");
		return 3;
	}

	return 0;
}

int nova_pomdp_pbvi_uninitialize_successors(int *&d_successors)
{
	if (d_successors != nullptr) {
		if (cudaFree(d_successors) != cudaSuccess) {
			fprintf(stderr, "Error[nova_pomdp_pbvi_uninitialize_successors]: %s",
					"Failed to allocate device-side memory for the successor states.");
			return 1;
		}
	}
	d_successors = nullptr;

	return 0;
}
