/*
* Copyright (c) 2010 Oh Kang Kwon. Released under GNU Public Licence Version 3.
Copyright (c) 2010 Mark Joshi. Released under GNU Public Licence Version 3.

*/

// Non CUDA specific code borrowed from quantlib...
//
// Copyright (C) 2003 Ferdinando Ametrano
// Copyright (C) 2006 StatPro Italia srl
// 
// This file is part of QuantLib, a free-software/open-source library
// for financial quantitative analysts and developers - http://quantlib.org/

// QuantLib is free software: you can redistribute it and/or modify it
// under the terms of the QuantLib license.  You should have received a
// copy of the license along with this program; if not, please email
// <quantlib-dev@lists.sf.net>. The license is also available online at
// <http://quantlib.org/license.shtml>.
// 
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the license for more details.

// ===========================================================================
// NOTE: The following copyright notice applies to the original code,
//
// Copyright (C) 2002 Peter J‰ckel "Monte Carlo Methods in Finance".
// All rights reserved.
//
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// ===========================================================================

#ifndef BROWNIAN_BRIDGE
#define BROWNIAN_BRIDGE
#include <gold/pragmas.h> 
#include <vector>
#include <thrust/device_vector.h>

/*
class to do Brownian bridge on GPU and CPU

output in cube of form steps,factors, paths 


*/

class brownian_bridge
{

public:
    brownian_bridge(int size, const std::vector<int>& allocation_indices);

    void transform(thrust::device_vector<float>& dev_input, 
                   thrust::device_vector<float>& dev_output,
                   int paths, 
                   int steps, 
                   int factors);

    void transform(std::vector<float>& input_vec,  
                   std::vector<float>& output_vec,  
                   int paths, 
                   int steps, 
                   int factors);

    void transform(float* input, float* output,  int paths, int steps, int factors, bool useDevice);  
private:
    void transformDevice(float* d_input, float* d_output,  int paths, int steps, int factors);
    void transformCPU(float* input_cpu, float* output_cpu,  int paths, int steps, int factors); 
   
    void initialize();
    void compute_kernel_configuration(int paths, int factors, int s_mem);
    void initDevice(); 

    int size_;
    std::vector<float> t_;
    std::vector<float> sqrtdt_;
    std::vector<int> bridgeIndex_, leftIndex_, rightIndex_;
    std::vector<float> leftWeight_, rightWeight_, stdDev_;
    std::vector<float> coefficient_;

    std::vector<int> allocation_indices_;

    dim3 dimGrid_;
    dim3 dimBlock_;

    bool deviceInitted_;
    thrust::device_vector<float> t_dev_;
    thrust::device_vector<float> sqrtdt_dev_;
    thrust::device_vector<int> bridgeIndex_dev_, leftIndex_dev_, rightIndex_dev_;
    thrust::device_vector<float> leftWeight_dev_, rightWeight_dev_, stdDev_dev_;
    thrust::device_vector<float> coefficient_dev_;

    thrust::device_vector<int> indices_dev_;

};

#endif
