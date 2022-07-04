//
//                                  cash_flow_product_equity_one_dim_gpu.h

//
//


// (c) Mark Joshi 2010,2012
// This code is released under the GNU public licence version 3

extern "C"
void payoffGeneratorCallerRatchetPut(float* genFlows,  float* aux_data, int paths, int numberSteps,
                                                      float* stockPaths);
