
//
//                                  cash_flow_product_equity_one_dim_main.cu
//
//


// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3
#include <cash_flow_product_equity_one_dim_main.h>
#include <cash_flow_product_equity_one_dim_gpu.h>
#include <thrust/device_vector.h>

void payoffGeneratorCallerRatchetPut_main(thrust::device_vector<float>& genFlows,  
                                          thrust::device_vector<float>& aux_data, 
                                          int paths, 
                                          int numberSteps,
                                          thrust::device_vector<float>& stockPaths
                                          )
{

    payoffGeneratorCallerRatchetPut( thrust::raw_pointer_cast(& genFlows[0]),  thrust::raw_pointer_cast(&aux_data[0]),
        paths, 
        numberSteps,
        thrust::raw_pointer_cast(& stockPaths[0]));

}
