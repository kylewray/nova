/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2014 Kyle Wray
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


#include "../include/nova_cuda.h"

#include <math.h>

#include <iostream> // ToDo: Remove me.

// This is not C++0x, unfortunately.
#define nullptr NULL

// This is determined by hardware, so what is below is a 'safe' guess. If this is
// off, the program might return 'nan' or 'inf'.
#define FLT_MAX 1e+35

__global__ void bellman_update(unsigned int n, unsigned int m, const float *T,
		const float *R, float gamma, const float *V, float *VPrime, unsigned int *pi)
{
	// The current state as a function of the blocks and threads.
	int s;

	// The intermediate Q(s, a) value.
	float Qsa;

	// The 1-d index version of the 3-d arrays in the innermost loop.
	int k;

	// Compute the index of the state. Return if it is beyond the state.
	s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= n) {
		return;
	}

	// Nvidia GPUs follow IEEE floating point standards, so this should be safe.
	VPrime[s] = -FLT_MAX;

	// Compute max_{a in A} Q(s, a).
	for (int a = 0; a < m; a++) {
		// Compute Q(s, a) for this action.
		Qsa = 0.0f;
		for (int sp = 0; sp < n; sp++) {
			k = s * m * n + a * n + sp;
			Qsa += T[k] * (R[k] + gamma * V[sp]);
		}

		if (a == 0 || Qsa > VPrime[s]) {
			VPrime[s] = Qsa;
			pi[s] = a;
		}
	}
}

__global__ void bellman_update_restricted_actions(unsigned int n, unsigned int m,
		const bool *A, const float *T, const float *R, float gamma, const float *V,
		float *VPrime, unsigned int *pi)
{
	// The current state as a function of the blocks and threads.
	int s;

	// The intermediate Q(s, a) value.
	float Qsa;

	// The 1-d index version of the 3-d arrays in the innermost loop.
	int k;

	// Compute the index of the state. Return if it is beyond the state.
	s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= n) {
		return;
	}

	// Nvidia GPUs follow IEEE floating point standards, so this should be safe.
	VPrime[s] = -FLT_MAX;

	// Compute max_{a in A} Q(s, a).
	for (int a = 0; a < m; a++) {
		// Skip this action if it is locked.
		if (!A[s * m + a]) {
			continue;
		}

		// Compute Q(s, a) for this action.
		Qsa = 0.0f;
		for (int sp = 0; sp < n; sp++) {
			k = s * m * n + a * n + sp;
			Qsa += T[k] * (R[k] + gamma * V[sp]);
		}

		if (a == 0 || Qsa > VPrime[s]) {
			VPrime[s] = Qsa;
			pi[s] = a;
		}
	}
}

int value_iteration(unsigned int n, unsigned int m, const float *T, const float *R,
		float Rmax, float gamma, float epsilon, float *V, unsigned int *pi,
		unsigned int numBlocks, unsigned int numThreads)
{
	// The device pointers for the MDP: T and R.
	float *d_T;
	float *d_R;

	// The host and device pointers for the value functions: V and VPrime.
	float *d_V;
	float *d_VPrime;

	// The device pointer for the final policy: pi.
	unsigned int *d_pi;

	// First, ensure data is valid.
	if (n == 0 || m == 0 || T == nullptr || R == nullptr ||
			gamma < 0.0f || gamma >= 1.0f || pi == nullptr) {
		return -1;
	}

	// Also ensure that there are enough blocks and threads to run the program.
	if (numBlocks * numThreads < n) {
		return -2;
	}

	// Next, determine how many iterations it will have to run.
	int iterations = 10 * (int)std::ceil(std::log(2.0f * Rmax / (epsilon * (1.0 - gamma)) / std::log(1.0 / gamma)));

	// Allocate the device-side memory.
	cudaMalloc(&d_T, n * m * n * sizeof(float));
	cudaMalloc(&d_R, n * m * n * sizeof(float));

	cudaMalloc(&d_V, n * sizeof(float));
	cudaMalloc(&d_VPrime, n * sizeof(float));

	cudaMalloc(&d_pi, n * sizeof(unsigned int));

	// Ensure that V and pi are initialized to zero to start.
	for (int s = 0; s < n; s++) {
		V[s] = 0.0f;
		pi[s] = 0;
	}

	// Copy the data from host to device.
	cudaMemcpy(d_T, T, n * m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, R, n * m * n * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_V, V, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_VPrime, V, n * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_pi, pi, n * sizeof(float), cudaMemcpyHostToDevice);

	// Execute value iteration for these number of iterations. For each iteration, however,
	// we will run the state updates in parallel.
	for (int i = 0; i < iterations; i++) {
//		printf("Iteration %d / %d\n", i, iterations);
//		printf("Blocks: %d\nThreads: %d\nGamma: %f\nn: %d\nm: %d\n", numBlocks, numThreads, gamma, n, m);

		if (i % 2 == 0) {
			bellman_update<<< numBlocks, numThreads >>>(n, m, d_T, d_R, gamma, d_V, d_VPrime, d_pi);
		} else {
			bellman_update<<< numBlocks, numThreads >>>(n, m, d_T, d_R, gamma, d_VPrime, d_V, d_pi);
		}
	}

	// Copy the final result, both V and pi, from device to host.
	if (iterations % 2 == 1) {
		cudaMemcpy(V, d_V, n * sizeof(float), cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(V, d_VPrime, n * sizeof(float), cudaMemcpyDeviceToHost);
	}
	cudaMemcpy(pi, d_pi, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

//	for (int s = 0; s < n; s++) {
//		printf("V[%d] =   %f\t", s, V[s]);
//		if (s % 8 == 7) {
//			printf("\n");
//		}
//	}

	// Free the device-side memory.
	cudaFree(d_T);
	cudaFree(d_R);

	cudaFree(d_V);
	cudaFree(d_VPrime);

	cudaFree(d_pi);

	return 0;
}

int value_iteration_restricted_actions(unsigned int n, unsigned int m, const bool *A,
		const float *T, const float *R, float Rmax, float gamma, float epsilon, float *V,
		unsigned int *pi, unsigned int numBlocks, unsigned int numThreads)
{
	// The device pointers for the MDP: A, T, and R.
	bool *d_A;
	float *d_T;
	float *d_R;

	// The host and device pointers for the value functions: V and VPrime.
	float *d_V;
	float *d_VPrime;

	// The device pointer for the final policy: pi.
	unsigned int *d_pi;

	// First, ensure data is valid.
	if (n == 0 || m == 0 || T == nullptr || R == nullptr ||
			gamma < 0.0f || gamma >= 1.0f || pi == nullptr) {
		return -1;
	}

	// Also ensure that there are enough blocks and threads to run the program.
	if (numBlocks * numThreads < n) {
		return -2;
	}

	// Next, determine how many iterations it will have to run.
	int iterations = 10 * (int)std::ceil(std::log(2.0f * Rmax / (epsilon * (1.0 - gamma)) / std::log(1.0 / gamma)));

	// Allocate the device-side memory.
	cudaMalloc(&d_A, n * m * sizeof(bool));
	cudaMalloc(&d_T, n * m * n * sizeof(float));
	cudaMalloc(&d_R, n * m * n * sizeof(float));

	cudaMalloc(&d_V, n * sizeof(float));
	cudaMalloc(&d_VPrime, n * sizeof(float));

	cudaMalloc(&d_pi, n * sizeof(unsigned int));

	// Ensure that V and pi are initialized to zero to start.
	for (int s = 0; s < n; s++) {
		V[s] = 0.0f;
		pi[s] = 0;
	}

	// Copy the data from host to device.
	cudaMemcpy(d_A, A, n * m * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, T, n * m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, R, n * m * n * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_V, V, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_VPrime, V, n * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_pi, pi, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

	// Execute value iteration for these number of iterations. For each iteration, however,
	// we will run the state updates in parallel.
	for (int i = 0; i < iterations; i++) {
//		printf("Iteration %d / %d\n", i, iterations);
//		printf("Blocks: %d\nThreads: %d\nGamma: %f\nn: %d\nm: %d\n", numBlocks, numThreads, gamma, n, m);

		if (i % 2 == 0) {
			bellman_update_restricted_actions<<< numBlocks, numThreads >>>(n, m, d_A, d_T, d_R, gamma, d_V, d_VPrime, d_pi);
		} else {
			bellman_update_restricted_actions<<< numBlocks, numThreads >>>(n, m, d_A, d_T, d_R, gamma, d_VPrime, d_V, d_pi);
		}
	}

	// Copy the final result, both V and pi, from device to host.
	if (iterations % 2 == 1) {
		cudaMemcpy(V, d_V, n * sizeof(float), cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(V, d_VPrime, n * sizeof(float), cudaMemcpyDeviceToHost);
	}
	cudaMemcpy(pi, d_pi, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

//	for (int s = 0; s < n; s++) {
//		printf("V[%d] =   %f\t", s, V[s]);
//		if (s % 8 == 7) {
//			printf("\n");
//		}
//	}

	// Free the device-side memory.
	cudaFree(d_A);
	cudaFree(d_T);
	cudaFree(d_R);

	cudaFree(d_V);
	cudaFree(d_VPrime);

	cudaFree(d_pi);

	return 0;
}
