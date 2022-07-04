//
//                                  cashFlowGeneration_product_gpu.cu
//
//


// (c) Mark Joshi 2010,2011,2013
// This code is released under the GNU public licence version 3
#include <cashFlowGeneration_product_gpu.h>


extern "C"
void cashFlowGeneratorCallerTARN(float* genFlows1, float* genFlows2, float* aux_data, int paths, int numberSteps,
                                                      float* rates1, 
                                                      float* rates2, 
                                                      float* rates3, 
                                                      float* forwards, 
                                                      float* discRatios)

{
    cashFlowGenerator<TARN>(genFlows1,genFlows2,aux_data,paths,numberSteps,rates1,rates2,rates3,forwards,discRatios);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



extern "C"
void cashFlowGeneratorCallerSwap(float* genFlows1, float* genFlows2, float* aux_data, int paths, int numberSteps,
                                                      float* rates1, 
                                                      float* rates2, 
                                                      float* rates3, 
                                                      float* forwards, 
                                                      float* discRatios)

{
    cashFlowGenerator<Swap>(genFlows1,genFlows2,aux_data,paths,numberSteps,rates1,rates2,rates3,forwards,discRatios);
}
