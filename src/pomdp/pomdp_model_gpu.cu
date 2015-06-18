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


#include "pomdp_model_gpu.h"
#include "error_codes.h"

#include <stdio.h>


int pomdp_initialize_successors_gpu(POMDP *pomdp)
{
    // Ensure the data is valid.
    if (pomdp->n == 0 || pomdp->m == 0 || pomdp->ns == 0 || pomdp->S == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_successors_gpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&pomdp->d_S, pomdp->n * pomdp->m * pomdp->ns * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_successors_gpu]: %s\n",
                "Failed to allocate device-side memory for the successor states.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(pomdp->d_S, pomdp->S, pomdp->n * pomdp->m * pomdp->ns * sizeof(int),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_successors_gpu]: %s\n",
                "Failed to copy memory from host to device for the successor states.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int pomdp_uninitialize_successors_gpu(POMDP *pomdp)
{
    if (pomdp->d_S != nullptr) {
        if (cudaFree(pomdp->d_S) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_successors_gpu]: %s\n",
                    "Failed to allocate device-side memory for the successor states.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_S = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_initialize_state_transitions_gpu(POMDP *pomdp)
{
    // Ensure the data is valid.
    if (pomdp->n == 0 || pomdp->m == 0 || pomdp->ns == 0 || pomdp->T == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_state_transitions_gpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&pomdp->d_T, pomdp->n * pomdp->m * pomdp->ns * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_state_transitions_gpu]: %s\n",
                "Failed to allocate device-side memory for the state transitions.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(pomdp->d_T, pomdp->T, pomdp->n * pomdp->m * pomdp->ns * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[nova_pomdp_pbvi_initialize_state_transitions]: %s\n",
                "Failed to copy memory from host to device for the state transitions.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int pomdp_uninitialize_state_transitions_gpu(POMDP *pomdp)
{
    if (pomdp->d_T != nullptr) {
        if (cudaFree(pomdp->d_T) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_state_transitions_gpu]: %s\n",
                    "Failed to allocate device-side memory for the state transitions.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_T = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_initialize_observation_transitions_gpu(POMDP *pomdp)
{
    // Ensure the data is valid.
    if (pomdp->n == 0 || pomdp->m == 0 || pomdp->z == 0 || pomdp->O == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_observation_transitions_gpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&pomdp->d_O, pomdp->m * pomdp->n * pomdp->z * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_observation_transitions_gpu]: %s\n",
                "Failed to allocate device-side memory for the observation transitions.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(pomdp->d_O, pomdp->O, pomdp->m * pomdp->n * pomdp->z * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_observation_transitions_gpu]: %s\n",
                "Failed to copy memory from host to device for the observation transitions.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int pomdp_uninitialize_observation_transitions_gpu(POMDP *pomdp)
{
    if (pomdp->d_O != nullptr) {
        if (cudaFree(pomdp->d_O) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_observation_transitions_gpu]: %s\n",
                    "Failed to allocate device-side memory for the observation transitions.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_O = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_initialize_rewards_gpu(POMDP *pomdp)
{
    // Ensure the data is valid.
    if (pomdp->n == 0 || pomdp->m == 0 || pomdp->R == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_rewards_gpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&pomdp->d_R, pomdp->n * pomdp->m * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_rewards_gpu]: %s\n",
                "Failed to allocate device-side memory for the rewards.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(pomdp->d_R, pomdp->R, pomdp->n * pomdp->m * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_rewards_gpu]: %s\n",
                "Failed to copy memory from host to device for the rewards.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int pomdp_uninitialize_rewards_gpu(POMDP *pomdp)
{
    if (pomdp->d_R != nullptr) {
        if (cudaFree(pomdp->d_R) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_rewards_gpu]: %s\n",
                    "Failed to allocate device-side memory for the rewards.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_R = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_initialize_nonzero_beliefs_gpu(POMDP *pomdp)
{
    // Ensure the data is valid.
    if (pomdp->r == 0 || pomdp->rz == 0 || pomdp->Z == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_nonzero_beliefs_gpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&pomdp->d_Z, pomdp->r * pomdp->rz * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_nonzero_beliefs_gpu]: %s\n",
                "Failed to allocate device-side memory for the non-zero belief states.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(pomdp->d_Z, pomdp->Z, pomdp->r * pomdp->rz * sizeof(int),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_nonzero_beliefs_gpu]: %s\n",
                "Failed to copy memory from host to device for the non-zero belief states.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int pomdp_uninitialize_nonzero_beliefs_gpu(POMDP *pomdp)
{
    if (pomdp->d_Z != nullptr) {
        if (cudaFree(pomdp->d_Z) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_nonzero_beliefs_gpu]: %s\n",
                    "Failed to allocate device-side memory for the non-zero belief states.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_Z = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_initialize_belief_points_gpu(POMDP *pomdp)
{
    // Ensure the data is valid.
    if (pomdp->r == 0 || pomdp->rz == 0 || pomdp->B == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_belief_points_gpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&pomdp->d_B, pomdp->r * pomdp->rz * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_belief_points_gpu]: %s\n",
                "Failed to allocate device-side memory for the belief points.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(pomdp->d_B, pomdp->B, pomdp->r * pomdp->rz * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_belief_points_gpu]: %s\n",
                "Failed to copy memory from host to device for the belief points.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}
 

int pomdp_uninitialize_belief_points_gpu(POMDP *pomdp)
{
    if (pomdp->d_B != nullptr) {
        if (cudaFree(pomdp->d_B) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_belief_points_gpu]: %s\n",
                    "Failed to allocate device-side memory for the belief points.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    pomdp->d_B = nullptr;

    return NOVA_SUCCESS;
}

