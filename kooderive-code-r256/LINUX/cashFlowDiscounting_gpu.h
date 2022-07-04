//
//
//                      cashFlowDiscounting_gpu.h
//
//
//
// (c) Mark Joshi 2010,2011
// This code is released under the GNU public licence version 3

#ifndef cash_flow_discounting_gpu_h
#define cash_flow_discounting_gpu_h


extern"C"
void cashFlowDiscounting_gpu(
                             int* firstIndex, 
                             int* secondIndex,
                             float* theta, 
                             float* discountRatios, 
                             float* genFlows, 
                             float* numeraireValues,
                             int paths, 
                             int numberSteps, 
                             bool useTextures,
							 bool useShared,
                             float* discountedFlows, // output
                             float* summedDiscountedFlows); // output

extern"C"
void cashFlowDiscounting_partial_gpu(int numberFlowsPerPath,
                                     int* generationIndices_dev,
                                     int* firstIndex_dev, 
                                     int* secondIndex_dev,
                                     float* theta_dev, 
                                     int* targetIndices_dev,
                                     float* discountRatios_dev, 
                                     float* genFlows_dev, 
                                     float* numeraireValues_dev,
                                     int paths, 
                                     int numberSteps, 
                                     float* discountedFlows_dev);


#endif
