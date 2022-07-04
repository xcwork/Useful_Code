// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3
// note derived from code provided by NVIDIA 

#ifndef OUTER_PRODUCT_MAIN_H
#define OUTER_PRODUCT_MAIN_H

#include <thrust/device_vector.h>



void reduceOuterProduct_main(int paths,
                            int row_size,
                            int col_size,
                            int threads, 
                            int blocks, 
							bool singleKernel,
                            thrust::device_vector<float>& input_data_global1, 
                            thrust::device_vector<float>& input_data_global2, 
                            thrust::device_vector<float>& answer_global,
                            thrust::device_vector<float>& workspace_data_global);

void reduceOuterProductSymmetric_main(int paths,
                            int row_size,
                            int threads, 
                            int blocks, 
                            thrust::device_vector<float>& input_data_global, 
                            thrust::device_vector<float>& answer_global,
                            thrust::device_vector<float>& workspace_data_global);


void reduceOuterProductSymmetricCublas_main(int paths,
                                            int row_size,
                                            float alpha,
                                            float beta,
                                            thrust::device_vector<float>& input_data_global, 
                                            thrust::device_vector<float>& answer_global);

extern "C"
void reduceOuterProductcublas_main(int paths,
						       int row_size,
                               int col_size,
									 float alpha,
                                     float beta,
									 thrust::device_vector<float>&  input_data_global1,
                                     thrust::device_vector<float>&  input_data_global2,
									 thrust::device_vector<float>&  answer_global);


double PointwiseProduct_main(int blocks,
						  int threads,
						  int paths,
						  thrust::device_vector<float>& in1_dev,
						  thrust::device_vector<float>& in2_dev,
						  thrust::device_vector<float>& out_dev);

#endif
