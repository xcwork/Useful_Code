//
//
//                              ComparisonCheck.h
//
//

#ifndef COMPARISON_CHECK_H
#define COMPARISON_CHECK_H
#include <thrust/device_vector.h>
#include <gold/MatrixFacade.h>
#include <gold/math/fp_utilities.h>

int numberMismatches(const MatrixFacade<float>& cpuData_matrix, const thrust::device_vector<float>& gpuData_device, double tolerance, bool dumpData=false);

template<class S, class T>
int numberMismatches(const std::vector<S>& cpuData_vec, const thrust::device_vector<T>& gpuData_device, double tolerance, bool dumpData)
{
    int err=0; 

    thrust::host_vector<T> gpuData_host(gpuData_device);


    for (size_t i =0; i < cpuData_vec.size(); ++i)
    {
        S cpuVal  = cpuData_vec[i];
        S gpuVal =  static_cast<S>(gpuData_host[i]);

        if (fp_isnan(cpuVal) || fp_isnan(gpuVal))
        {
            ++err;
            std::cout << i << "," << cpuVal  << "," << gpuVal << "\n";

        }
        else
            if (fabs(cpuVal -gpuVal) > tolerance)
                ++err;

        if (dumpData)
            std::cout << i << "," << cpuVal  << "," <<  gpuVal<< "\n";
    }

    return err;
}


#endif

