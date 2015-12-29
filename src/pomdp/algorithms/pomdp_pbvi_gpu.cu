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


#include "algorithms/pomdp_pbvi_gpu.h"
#include "utilities/pomdp_model_gpu.h"
#include "error_codes.h"
#include "constants.h"

#include <stdio.h>

namespace nova {

__global__ void pomdp_pbvi_initialize_alphaBA_gpu(unsigned int n, unsigned int m,
    unsigned int r, const float *R, float *alphaBA)
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


__global__ void pomdp_pbvi_compute_alphaBA_gpu(unsigned int n, unsigned int ns, unsigned int m,
        unsigned int z, unsigned int r, unsigned int rz, float gamma,
        const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
        const float *Gamma, float *alphaBA)
{
    // Since float and unsigned int are 4 bytes each, and we need each array to be the size of
    // the number of threads, we will need to call this with:
    // sizeof(float) * numThreads + sizeof(unsigned int) * numThreads.
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

        for (unsigned int i = 0; i < rz; i++) {
            int s = Z[beliefIndex * rz + i];
            if (s < 0) {
                break;
            }

            // We compute the value of this state in the alpha-vector, then multiply it by the
            // belief, and add it to the current dot product value for this alpha-vector.
            float value = 0.0f;
            for (unsigned int j = 0; j < ns; j++) {
                int sp = S[s * m * ns + action * ns + j];
                if (sp < 0) {
                    break;
                }
                value += T[s * m * ns + action * ns + j] *
                            O[action * n * z + sp * z + observation] *
                            Gamma[alphaIndex * n + sp];
            }

            __syncthreads();

            value *= gamma;

            alphaDotBeta += value * B[beliefIndex * rz + i];
        }

        __syncthreads();

        // Store the maximal value and index.
        if (alphaIndex == threadIdx.x || alphaDotBeta > maxAlphaDotBeta[threadIdx.x]) {
            maxAlphaDotBeta[threadIdx.x] = alphaDotBeta;
            maxAlphaIndex[threadIdx.x] = alphaIndex;
        }
    }

    // Note: The above code essentially does the first add during load. It takes care of *all*
    // the other elements *outside* the number of threads we have. In other words, starting here,
    // we already have computed part of the maxAlphaDotBeta and maxAlphaIndex; we just need to
    // finish the rest quickly, using a reduction.
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
        // We compute the value of this state in the alpha-vector, then multiply it by the belief,
        // and add it to the current dot product value for this alpha-vector.
        float value = 0.0f;
        for (unsigned int i = 0; i < ns; i++) {
            int sp = S[s * m * ns + action * ns + i];
            if (sp < 0) {
                break;
            }
            // Note: maxAlphaIndex[0] holds the maximal index value computed from the reduction.
            value += T[s * m * ns + action * ns + i] *
                        O[action * n * z + sp * z + observation] *
                        Gamma[maxAlphaIndex[0] * n + sp];
        }

        __syncthreads();

        alphaBA[beliefIndex * m * n + action * n + s] += gamma * value;
    }
}


__global__ void pomdp_pbvi_update_step_gpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
        unsigned int r, unsigned int rz, float gamma,
        const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
        const float *Gamma,
        float *alphaBA, float *GammaPrime, unsigned int *piPrime)
{
    // Each block will run a different belief. Our overall goal: Compute the value
    // of GammaPrime[beliefIndex * n + ???] and piPrime[beliefIndex].
    unsigned int beliefIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (beliefIndex >= r) {
        return;
    }

    // We want to find the action that maximizes the value, store it in piPrime, as well as
    // its alpha-vector GammaPrime.
    float maxActionValue = FLT_MIN;

    for (unsigned int action = 0; action < m; action++) {
        // Only execute if the action is available.
        //if (available[beliefIndex * m + action]) {

        // The potential alpha-vector has been computed, so compute the value with respect
        // to the belief state.
        float actionValue = 0.0f;
        for (unsigned int i = 0; i < rz; i++) {
            int s = Z[beliefIndex * rz + i];
            if (s < 0) {
                break;
            }
            actionValue += alphaBA[beliefIndex * m * n + action * n + s] * B[beliefIndex * rz + i];
        }

        // If this was larger, then overwrite piPrime and GammaPrime's values.
        if (actionValue > maxActionValue) {
            maxActionValue = actionValue;

            piPrime[beliefIndex] = action;
        }

        //}
        //__syncthreads();
    }

    memcpy(&GammaPrime[beliefIndex * n], &alphaBA[beliefIndex * m * n + piPrime[beliefIndex] * n], n * sizeof(float));
    //for (unsigned int s = 0; s < n; s++) {
    //    GammaPrime[beliefIndex * n + s] = alphaBA[beliefIndex * m * n + piPrime[beliefIndex] * n + s];
    //}
}


int pomdp_pbvi_complete_gpu(POMDP *pomdp, unsigned int numThreads, const float *initialGamma, POMDPAlphaVectors *&policy)
{
    int result;

    result = pomdp_initialize_successors_gpu(pomdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_state_transitions_gpu(pomdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_observation_transitions_gpu(pomdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_rewards_gpu(pomdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_nonzero_beliefs_gpu(pomdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_belief_points_gpu(pomdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    result = pomdp_pbvi_execute_gpu(pomdp, numThreads, initialGamma, policy);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    result = NOVA_SUCCESS;
    if (pomdp_uninitialize_successors_gpu(pomdp) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_state_transitions_gpu(pomdp) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_observation_transitions_gpu(pomdp) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_rewards_gpu(pomdp) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_nonzero_beliefs_gpu(pomdp) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_belief_points_gpu(pomdp) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }

    return result;
}


int pomdp_pbvi_initialize_gpu(POMDP *pomdp, const float *initialGamma)
{
    // Reset the current horizon.
    pomdp->currentHorizon = 0;

    // Create the device-side Gamma.
    if (cudaMalloc(&pomdp->d_Gamma, pomdp->r * pomdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for Gamma.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMemcpy(pomdp->d_Gamma, initialGamma, pomdp->r * pomdp->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to copy memory from host to device for Gamma.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    if (cudaMalloc(&pomdp->d_GammaPrime, pomdp->r * pomdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for Gamma (prime).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMemcpy(pomdp->d_GammaPrime, initialGamma, pomdp->r * pomdp->n * sizeof(float),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to copy memory from host to device for Gamma (prime).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    // Create the device-side pi.
    if (cudaMalloc(&pomdp->d_pi, pomdp->r * sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for pi.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Create the device-side memory for the intermediate variable alphaBA.
    if (cudaMalloc(&pomdp->d_alphaBA, pomdp->r * pomdp->m * pomdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for alphaBA.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_execute_gpu(POMDP *pomdp, unsigned int numThreads, const float *initialGamma, POMDPAlphaVectors *&policy)
{
    // The result from calling other functions.
    int result;

    // Ensure the data is valid.
    if (pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 || pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->d_S == nullptr || pomdp->d_T == nullptr || pomdp->d_O == nullptr || pomdp->d_R == nullptr ||
            pomdp->d_Z == nullptr || pomdp->d_B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            initialGamma == nullptr || policy != nullptr) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Ensure threads are correct.
    if (numThreads % 32 != 0) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s\n", "Invalid number of threads.");
        return NOVA_ERROR_INVALID_CUDA_PARAM;
    }

    result = pomdp_pbvi_initialize_gpu(pomdp, initialGamma);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    // For each of the updates, run PBVI. Note that the currentHorizon is initialized to zero
    // above, and is updated in the update function below.
    while (pomdp->currentHorizon < pomdp->horizon) {
        //printf("PBVI (GPU Version) -- Iteration %i of %i\n", pomdp->currentHorizon, pomdp->horizon);

        result = pomdp_pbvi_update_gpu(pomdp, numThreads);
        if (result != NOVA_SUCCESS) {
            return result;
        }
    }

    result = pomdp_pbvi_get_policy_gpu(pomdp, policy);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    result = pomdp_pbvi_uninitialize_gpu(pomdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_uninitialize_gpu(POMDP *pomdp)
{
    int result;

    result = NOVA_SUCCESS;

    // Reset the current horizon.
    pomdp->currentHorizon = 0;

    if (pomdp->d_Gamma != nullptr) {
        if (cudaFree(pomdp->d_Gamma) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s\n",
                    "Failed to allocate device-side memory for the Gamma (the alpha-vectors).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_Gamma = nullptr;

    if (pomdp->d_GammaPrime != nullptr) {
        if (cudaFree(pomdp->d_GammaPrime) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s\n",
                    "Failed to allocate device-side memory for the GammaPrime (the alpha-vectors' copy).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_GammaPrime = nullptr;

    if (pomdp->d_pi != nullptr) {
        if (cudaFree(pomdp->d_pi) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s\n",
                    "Failed to allocate device-side memory for the pi (the policy).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_pi = nullptr;

    if (pomdp->d_alphaBA != nullptr) {
        if (cudaFree(pomdp->d_alphaBA) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s\n",
                    "Failed to allocate device-side memory for alphaBA (alpha-vector collection).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_alphaBA = nullptr;

    return result;
}


int pomdp_pbvi_update_gpu(POMDP *pomdp, unsigned int numThreads)
{
    // The number of blocks in the main CUDA kernel call.
    int numBlocks;

    pomdp_pbvi_initialize_alphaBA_gpu<<< dim3(pomdp->r, pomdp->m, 1), numThreads >>>(
                                            pomdp->n, pomdp->m, pomdp->r, pomdp->d_R, pomdp->d_alphaBA);

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to execute the 'initialization of alphaBA' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                    "Failed to synchronize the device after 'initialization of alphaBA' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    pomdp_pbvi_compute_alphaBA_gpu<<< dim3(pomdp->r, pomdp->m, pomdp->z), numThreads,
                                    numThreads * sizeof(float) + numThreads * sizeof(unsigned int) >>>(
                            pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                            pomdp->d_S, pomdp->d_T, pomdp->d_O, pomdp->d_R, pomdp->d_Z, pomdp->d_B,
                            pomdp->d_Gamma,
                            pomdp->d_alphaBA);

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to execute the 'compute_alphaBA' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to synchronize the device after 'compute_alphaBA' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    // Compute the number of blocks.
    numBlocks = (unsigned int)((float)pomdp->r / (float)numThreads) + 1;

    // Execute a kernel for the first three stages of for-loops: B, A, Z, as a 3d-block,
    // and the 4th stage for-loop over Gamma as the threads.
    if (pomdp->currentHorizon % 2 == 0) {
        pomdp_pbvi_update_step_gpu<<< numBlocks, numThreads >>>(
                pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                pomdp->d_S, pomdp->d_T, pomdp->d_O, pomdp->d_R, pomdp->d_Z, pomdp->d_B,
                pomdp->d_Gamma,
                pomdp->d_alphaBA,
                pomdp->d_GammaPrime, pomdp->d_pi);
    } else {
        pomdp_pbvi_update_step_gpu<<< numBlocks, numThreads >>>(
                pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                pomdp->d_S, pomdp->d_T, pomdp->d_O, pomdp->d_R, pomdp->d_Z, pomdp->d_B,
                pomdp->d_GammaPrime,
                pomdp->d_alphaBA,
                pomdp->d_Gamma, pomdp->d_pi);
    }

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to execute the 'pomdp_pbvi_update_step_gpu' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s\n",
                        "Failed to synchronize the device after 'pomdp_pbvi_update_step_gpu' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    pomdp->currentHorizon++;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_get_policy_gpu(const POMDP *pomdp, POMDPAlphaVectors *&policy)
{
    if (policy != nullptr) {
        fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s\n", "Invalid arguments. Policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    policy = new POMDPAlphaVectors();

    policy->n = pomdp->n;
    policy->m = pomdp->m;
    policy->r = pomdp->r;

    policy->Gamma = new float[pomdp->r * pomdp->n];
    policy->pi = new unsigned int[pomdp->r];

    // Copy the final result of Gamma and pi to the variables provided, from device to host.
    // This assumes that the memory has been allocated for the variables provided.
    if (pomdp->currentHorizon % 2 == 0) {
        if (cudaMemcpy(policy->Gamma, pomdp->d_Gamma, pomdp->r * pomdp->n * sizeof(float),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s\n",
                    "Failed to copy memory from device to host for Gamma.");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    } else {
        if (cudaMemcpy(policy->Gamma, pomdp->d_GammaPrime, pomdp->r * pomdp->n * sizeof(float),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s\n",
                    "Failed to copy memory from device to host for Gamma (prime).");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    }

    if (cudaMemcpy(policy->pi, pomdp->d_pi, pomdp->r * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s\n",
                "Failed to copy memory from device to host for pi.");
        return NOVA_ERROR_MEMCPY_TO_HOST;
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

