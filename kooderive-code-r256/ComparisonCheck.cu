//
//
//                              ComparisonCheck.cu
//
//
#include <ComparisonCheck.h>
#include <gold/math/fp_utilities.h>
#include <cutil.h>

int numberMismatches(const MatrixFacade<float>& cpuData_matrix, const thrust::device_vector<float>& gpuData_device, double tolerance, bool dumpData)
{
    int err=0; 

    thrust::host_vector<float> gpuData_host(gpuData_device);
    MatrixFacade<float> gpuData_matrix(&gpuData_host[0],cpuData_matrix.rows(),cpuData_matrix.columns());


    for (int r=0; r < cpuData_matrix.rows(); ++r)
        for (int c=0; c < cpuData_matrix.columns(); ++c)
        {
            float cpuVal  = cpuData_matrix(r,c);
            float gpuVal =  gpuData_matrix(r,c);

            if (fp_isnan(cpuVal) || fp_isnan(gpuVal))
            {
                ++err;
                std::cout << r << "," << c << "," << cpuData_matrix(r,c)  << "," <<  gpuData_matrix(r,c) << "\n";

            }
            else
                if (fabs(cpuVal -gpuVal) > tolerance)
                    ++err;

            if (dumpData)
                std::cout << r << "," << c << "," << cpuData_matrix(r,c)  << "," <<  gpuData_matrix(r,c) << "\n";
        }

        return err;
}
