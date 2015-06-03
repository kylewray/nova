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


#include "mdp_vi.h"
#include "error_codes.h"

#include <cmath>
#include <stdio.h>

// This is determined by hardware, so what is below is a 'safe' guess. If this is
// off, the program might return 'nan' or 'inf'.
#define FLT_MAX 1e+35

__global__ void nova_mdp_bellman_update(unsigned int n, unsigned int m, unsigned int ns,
        const int *S, const float *T, const float *R, float gamma, const float *V, float *VPrime, unsigned int *pi)
{
    // The current state as a function of the blocks and threads.
    int s;

    // The intermediate Q(s, a) value.
    float Qsa;

    // The index within S and T (i.e., in n*s*ns).
    int index;

    // The true successor state index (in 0 to n-1), resolved using S.
    int spindex;

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
        Qsa = R[s * m + a];

        for (int sp = 0; sp < ns; sp++) {
            index = s * m * ns + a * ns + sp;

            spindex = S[index];
            if (spindex < 0) {
                break;
            }

            Qsa += gamma * T[index] * V[spindex];
        }

        __syncthreads();

        if (a == 0 || Qsa > VPrime[s]) {
            VPrime[s] = Qsa;
            pi[s] = a;
        }

        __syncthreads();
    }
}

int nova_mdp_vi(unsigned int n, unsigned int m, unsigned int ns,
                const int *S, const float *T, const float *R,
                float gamma, unsigned int horizon, unsigned int numThreads,
                float *V, unsigned int *pi)
{
    // The device pointers for the MDP: S, T, and R.
    int *d_S;
    float *d_T;
    float *d_R;

    // The host and device pointers for the value functions: V and VPrime.
    float *d_V;
    float *d_VPrime;

    // The device pointer for the final policy: pi.
    unsigned int *d_pi;

    // The number of blocks
    unsigned int numBlocks;

    // The final return message code, which is set during freeing memory at the end.
    unsigned int result;

    // First, ensure data is valid.
    if (n == 0 || m == 0 || ns == 0 ||
            S == nullptr || T == nullptr || R == nullptr ||
            gamma < 0.0f || gamma >= 1.0f || horizon < 1 ||
            V == nullptr || pi == nullptr) {
        return NOVA_ERROR_INVALID_DATA;
    }

    // Ensure threads are correct.
    if (numThreads % 32 != 0) {
        fprintf(stderr, "Error[value_iteration]: %s", "Invalid number of threads.");
        return NOVA_ERROR_INVALID_CUDA_PARAM;
    }

    numBlocks = (unsigned int)((float)n / (float)numThreads) + 1;

    // Allocate the device-side memory.
    if (cudaMalloc(&d_S, n * m * ns * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the successor states.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&d_T, n * m * ns * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the state transitions.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&d_R, n * m * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the rewards.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    if (cudaMalloc(&d_V, n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the value function.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&d_VPrime, n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the value function (prime).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    if (cudaMalloc(&d_pi, n * sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to allocate device-side memory for the policy (pi).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    /*
    // Assume that V and pi are initialized *properly* (either 0, or, with MPI, perhaps
    // with previous V values).

    for (int s = 0; s < n; s++) {
        V[s] = 0.0f;
        pi[s] = 0;
    }
    //*/

    // Copy the data from host to device.
    if (cudaMemcpy(d_S, S, n * m * ns * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the successors.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }
    if (cudaMemcpy(d_T, T, n * m * ns * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the state transitions.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }
    if (cudaMemcpy(d_R, R, n * m * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the rewards.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    if (cudaMemcpy(d_V, V, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the value function.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }
    if (cudaMemcpy(d_VPrime, V, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the value function (prime).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    if (cudaMemcpy(d_pi, pi, n * sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from host to device for the policy (pi).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    // Execute value iteration for these number of iterations. For each iteration, however,
    // we will run the state updates in parallel.
    for (int i = 0; i < horizon; i++) {
        printf("Iteration %d / %d\n", i, horizon);

        if (i % 2 == 0) {
            nova_mdp_bellman_update<<< numBlocks, numThreads >>>(n, m, ns, d_S, d_T, d_R, gamma, d_V, d_VPrime, d_pi);
        } else {
            nova_mdp_bellman_update<<< numBlocks, numThreads >>>(n, m, ns, d_S, d_T, d_R, gamma, d_VPrime, d_V, d_pi);
        }
    }

    // Copy the final result, both V and pi, from device to host.
    if (horizon % 2 == 1) {
        if (cudaMemcpy(V, d_V, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[value_iteration]: %s",
                    "Failed to copy memory from device to host for the value function.");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    } else {
        if (cudaMemcpy(V, d_VPrime, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[value_iteration]: %s",
                    "Failed to copy memory from device to host for the value function (prime).");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    }
    if (cudaMemcpy(pi, d_pi, n * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to copy memory from device to host for the policy (pi).");
        return NOVA_ERROR_MEMCPY_TO_HOST;
    }

    // Free the device-side memory. Note that we continue on error, since it is just cleaning
    // memory, but it is always nice to know if this failed anywhere.
    result = NOVA_SUCCESS;

    if (cudaFree(d_S) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the successor states.");
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (cudaFree(d_T) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the state transitions.");
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (cudaFree(d_R) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the rewards.");
        result = NOVA_ERROR_DEVICE_FREE;
    }

    if (cudaFree(d_V) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the value function.");
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (cudaFree(d_VPrime) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the value function (prime).");
        result = NOVA_ERROR_DEVICE_FREE;
    }

    if (cudaFree(d_pi) != cudaSuccess) {
        fprintf(stderr, "Error[value_iteration]: %s",
                "Failed to free memory from device for the policy (pi).");
        result = NOVA_ERROR_DEVICE_FREE;
    }

    return result;
}

