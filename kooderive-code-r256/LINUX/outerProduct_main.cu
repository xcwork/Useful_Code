// (c) Mark Joshi 2012, 2013
// This code is released under the GNU public licence version 3
// note derived from code provided by NVIDIA 


#include <outerProduct_main.h>
#include <outerProduct_gpu.h>

#include <math.h>
#include <vector>
#include <gold/Errors.h>
#include <gold/MatrixFacade.h>

void reduceOuterProduct_main(int paths,
                             int row_size,
                             int col_size,
                             int threads, 
                             int blocks, 
							 bool singleKernel,
                             thrust::device_vector<float>& input_data_global1, 
                             thrust::device_vector<float>& input_data_global2, 
                             thrust::device_vector<float>& answer_global,
                             thrust::device_vector<float>& workspace_data_global)
{
    if (threads==0)
        threads = 128;
    if (blocks ==0)
        blocks = 128;

    if (workspace_data_global.size() < static_cast<unsigned int>(blocks)*row_size*col_size)
        workspace_data_global.resize(blocks*row_size*col_size);

	if (singleKernel)
	{

		GenerateError("single kernel not currently supported.");

	}
	else
    reduceOuterProduct_gpu(paths,
        row_size,
        col_size,
        false,
        threads, 
        blocks, 
        thrust::raw_pointer_cast(&input_data_global1[0]), 
        thrust::raw_pointer_cast(&input_data_global2[0]), 
        thrust::raw_pointer_cast(&answer_global[0]),
        thrust::raw_pointer_cast(&workspace_data_global[0])
        );

	//	debugDumpMatrix(input_data_global1,"input_data_global1", row_size, paths);
	//	debugDumpMatrix(input_data_global2,"input_data_global2", col_size, paths);

	//	debugDumpCube(workspace_data_global,"workspace_data_global", row_size, col_size, blocks );



}


void reduceOuterProductSymmetric_main(int paths,
                                      int row_size,
                                      int threads, 
                                      int blocks, 
                                      thrust::device_vector<float>& input_data_global, 
                                      thrust::device_vector<float>& answer_global,
                                      thrust::device_vector<float>& workspace_data_global)
{
    if (threads==0)
        threads = 512;
    if (blocks ==0)
        blocks = 180;

    if (workspace_data_global.size() < static_cast<unsigned int>(blocks)*row_size*row_size)
        workspace_data_global.resize(blocks);

    reduceOuterProductSymmetric_gpu(paths,
        row_size,
        threads, 
        blocks, 
        thrust::raw_pointer_cast(&input_data_global[0]), 
        thrust::raw_pointer_cast(&answer_global[0]),
        thrust::raw_pointer_cast(&workspace_data_global[0])
        );



}
void reduceOuterProductSymmetricCublas_main(int paths,
                                            int row_size,
                                            float alpha,
                                            float beta,
                                            thrust::device_vector<float>& input_data_global, 
                                            thrust::device_vector<float>& answer_global)
{
    reduceOuterProductSymmetric_cublas( paths,
        row_size,
        alpha,
        beta,
        thrust::raw_pointer_cast(&input_data_global[0]), 
        thrust::raw_pointer_cast(&answer_global[0]));


}


void reduceOuterProductcublas_main(int paths,
						       int row_size,
                               int col_size,
									 float alpha,
                                     float beta,
									 thrust::device_vector<float>&  input_data_global1,
                                     thrust::device_vector<float>&  input_data_global2,
									 thrust::device_vector<float>&  answer_global)
{

    reduceOuterProduct_cublas(paths,
						 row_size,
                         col_size,
						 alpha,
                         beta,
						 thrust::raw_pointer_cast(&input_data_global1[0]),
                         thrust::raw_pointer_cast(&input_data_global2[0]), 
                         thrust::raw_pointer_cast(&answer_global[0]));


}
double PointwiseProduct_main(int blocks,
						  int threads,
						  int paths,
						  thrust::device_vector<float>& in1_dev,
						  thrust::device_vector<float>& in2_dev,
						  thrust::device_vector<float>& out_dev)
{
	return PointwiseProduct_gpu(blocks,
						  threads,
						  paths,
						   thrust::raw_pointer_cast(&in1_dev[0]),
						   thrust::raw_pointer_cast(&in2_dev[0]),
						   thrust::raw_pointer_cast(&out_dev[0]));
}
