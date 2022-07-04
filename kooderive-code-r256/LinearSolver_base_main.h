
#ifndef LINEARSOLVER_BASE_MAIN_H
#define LINEARSOLVER_BASE_MAIN_H
#include <thrust/device_vector.h>


// solve Ax =y, by taking ATA x= ATy and then using SVD
// abstract base case to allow for varying implementations
// inputs is AT not A since this is more natural for a lot of CUDA code as well as standard in BLAS routines
class LinearSolverBaseMain
{
public: 
	LinearSolverBaseMain(){}

	virtual void setDimensions(int rows, int columns)=0;

	virtual void solveSystem(thrust::device_vector<float>& AT, thrust::device_vector<float>& y, thrust::device_vector<float>&  )=0;

	virtual ~LinearSolverBaseMain(){}
};

#endif
