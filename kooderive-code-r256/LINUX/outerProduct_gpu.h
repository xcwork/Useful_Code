
//
//
//                                             outerProduct_gpu.h
//
// (c) Mark Joshi 2012,2013
// This code is released under the GNU public licence version 3

#ifndef OUTER_PRODUCT_GPU_H
#define OUTER_PRODUCT_GPU_H

void reduceOuterProduct_gpu(int paths,
                            int row_size,
                            int col_size,
                            bool symmetric,
                            int threads, 
                            int blocks, 
                            float* input_data_global1,
                            float* input_data_global2,
                            float* answer_global,
                            float* workspace_data_global);


void reducedOuterProduct_gpu(int paths,
									  int row_size,
									  int col_size,
									  int threads, 
									  int blocks, 
									  float* input_data_global1,
									  float* input_data_global2,
									  float* answer_global,
									  float* workspace_data_global);
void reduceOuterProductSymmetric_gpu(int paths,
									  int row_size,
									  int threads, 
									  int blocks, 
									  float* input_data_global1,
									  float* answer_global,
									  float* workspace_data_global);


void reduceOuterProductSymmetric_cublas(int paths,
									 int row_size,
									 float alpha,
                                     float beta,
									 float* input_data_global,
									 float* answer_global);


void reduceOuterProductSymmetric_cublas_initted(int paths,
									 int row_size,
									 float alpha,
                                     float beta,
									 float* input_data_global,
									 float* answer_global);


void reduceOuterProduct_cublas(int paths,
						       int row_size,
                               int col_size,
									 float alpha,
                                     float beta,
									 float* input_data_global1,
                                     float* input_data_global2,
									 float* answer_global);

void reduceOuterProduct_cublas_initted(int paths,
						       int row_size,
                               int col_size,
									 float alpha,
                                     float beta,
									 float* input_data_global1,
                                     float* input_data_global2,
									 float* answer_global);
/*
void reduceOuterProduct_cublas_initted(int paths,
						       int row_size,
                               int col_size,
									 double alpha,
                                     double beta,
									 double* input_data_global1,
                                     double* input_data_global2,
									 double* answer_global);

void reduceOuterProductSymmetric_cublas_initted(int paths,
									 int row_size,
									 double alpha,
                                     double beta,
									 double* input_data_global,
									 double* answer_global);
									 */


double PointwiseProduct_gpu(int blocks,
						  int threads,
						  int paths,
						  const float* in1_global,
						  const float* in2_global,
						  float* out_global);

double PointwiseProductSymmetricUpperTriangular_gpu(int blocks,
						  int threads,
						  int paths,
						  int rows,
						  const float* in_global,
						  float* out_global);


double PointwiseProductMultipleFirst_gpu(int blocks,
						  int threads,
						  int paths,
						  const float* in1_global,
						  const float* in2_global,
						  float* out_global,
						  int rows);
#endif
