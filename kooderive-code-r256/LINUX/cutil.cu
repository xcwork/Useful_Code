// cutil.cu

// (c) Mark Joshi 2013

// released under GPL v3

// derived from code carrying following licence

/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
Extract from EULA: 
Source Code:  Developer shall have the right to modify and create derivative works with the Source Code.  
Developer shall own any derivative works ("Derivatives") it creates to the Source Code, provided that
Developer uses the Materials in accordance with the terms and conditions of this Agreement.  Developer may
distribute the Derivatives, provided that all NVIDIA copyright notices and trademarks are propagated and 
used properly and the Derivatives include the following statement: 
"This software contains source code provided by NVIDIA Corporation."  


 */
#include <cutil.h>
#include <algorithm>
#include <stdio.h>

namespace
{
	// Beginning of GPU Architecture definitions
	int _ConvertSMVer2Cores(int major, int minor)
	{
		// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
		typedef struct
		{
			int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
			int Cores;
		} sSMtoCores;

		sSMtoCores nGpuArchCoresPerSM[] =
		{
			{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
			{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
			{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
			{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
			{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
			{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
			{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
			{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
			{   -1, -1 }
		};

		int index = 0;

		while (nGpuArchCoresPerSM[index].SM != -1)
		{
			if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
			{
				return nGpuArchCoresPerSM[index].Cores;
			}

			index++;
		}

		// If we don't find the values, we default use the previous one to run properly
		printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
		return nGpuArchCoresPerSM[7].Cores;
	}
	// end of GPU Architecture definitions
	int gpuGetMaxGflopsDeviceId()
	{
		int current_device     = 0, sm_per_multiproc  = 0;
		int max_compute_perf   = 0, max_perf_device   = 0;
		int device_count       = 0, best_SM_arch      = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceCount(&device_count);

		// Find the best major SM Architecture GPU device
		while (current_device < device_count)
		{
			cudaGetDeviceProperties(&deviceProp, current_device);

			// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
			if (deviceProp.computeMode != cudaComputeModeProhibited)
			{
				if (deviceProp.major > 0 && deviceProp.major < 9999)
				{
					best_SM_arch = std::max(best_SM_arch, deviceProp.major);
				}
			}

			current_device++;
		}

		// Find the best CUDA capable GPU device
		current_device = 0;

		while (current_device < device_count)
		{
			cudaGetDeviceProperties(&deviceProp, current_device);

			// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
			if (deviceProp.computeMode != cudaComputeModeProhibited)
			{
				if (deviceProp.major == 9999 && deviceProp.minor == 9999)
				{
					sm_per_multiproc = 1;
				}
				else
				{
					sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
				}

				int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

				if (compute_perf  > max_compute_perf)
				{
					// If we find GPU with SM major > 2, search only these
					if (best_SM_arch > 2)
					{
						// If our device==dest_SM_arch, choose this, or else pass
						if (deviceProp.major == best_SM_arch)
						{
							max_compute_perf  = compute_perf;
							max_perf_device   = current_device;
						}
					}
					else
					{
						max_compute_perf  = compute_perf;
						max_perf_device   = current_device;
					}
				}
			}

			++current_device;
		}

		return max_perf_device;
	}
}

int cutGetMaxGflopsDeviceId() {return gpuGetMaxGflopsDeviceId();}
