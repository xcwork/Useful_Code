
//
//
//                          early_exercise_value_generator_main.cu
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3
#include <early_exercise_value_generator_main.h>
#include <early_exercise_value_generator_gpu.h>


extern "C"
void adjoinExerciseValues_Bermudan_swaption_main(thrust::device_vector<float>& exerciseValues_device,  // output, one per exercise time per path
                               thrust::device_vector<float>& aux_float_data_device, // any float auxiliary data for the exercise value object 
                               thrust::device_vector<int>& aux_int_data_device, //  any int auxiliary data for the exercise value object 
                               thrust::device_vector<int>& exerciseIndices_device, // the indices of the exercise times amongst evolution times
                               thrust::device_vector<int>& exerciseIndicators_device, // boolean indicators of exercise amongst evolution times 
                               int pathsForOutput, // typically totalPaths
                               int outputOffset,
                               int batchPaths, 
                               int numberStepsAndRates, // assumed to be equal 
                               thrust::device_vector<float>& rates1_device, 
                               thrust::device_vector<float>& rates2_device, 
                               thrust::device_vector<float>& rates3_device, 
                               thrust::device_vector<float>& forwards_device, 
                               thrust::device_vector<float>& discRatios_device)
 {
     float tmp = aux_float_data_device[0];
     adjoinExerciseValues_Bermudan_swaption_gpu( thrust::raw_pointer_cast(&exerciseValues_device[0]),  // output, one per exercise time per path
                               thrust::raw_pointer_cast(&aux_float_data_device[0]), // any float auxiliary data for the exercise value object 
                               thrust::raw_pointer_cast(&aux_int_data_device[0]), //  any int auxiliary data for the exercise value object 
                               thrust::raw_pointer_cast(&exerciseIndices_device[0]), // the indices of the exercise times amongst evolution times
                               thrust::raw_pointer_cast(&exerciseIndicators_device[0]), // boolean indicators of exercise amongst evolution times 
                               exerciseIndices_device.size(),
                               pathsForOutput, // typically totalPaths
                               outputOffset,
                               batchPaths, 
                               numberStepsAndRates, // assumed to be equal 
                               thrust::raw_pointer_cast(&rates1_device[0]), 
                               thrust::raw_pointer_cast(&rates2_device[0]), 
                               thrust::raw_pointer_cast(&rates3_device[0]), 
                               thrust::raw_pointer_cast(&forwards_device[0]), 
                               thrust::raw_pointer_cast(& discRatios_device[0])
                          );
 }
