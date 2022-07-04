
#ifndef LINEARSOLVER_CONCRETE_MAIN_H
#define LINEARSOLVER_CONCRETE_MAIN_H
#include <thrust/device_vector.h>
#include <LinearSolver_base_main.h>
#include <reduction_thread_fence.h>
#include <gold/math/svd_gold.h>


// reduce to square system on gpu but do the solving on CPU
// main virtue is that removes dependence on external libraries
class LinearSolverHandCraftedAndGold : public 	LinearSolverBaseMain
{
public: 
	LinearSolverHandCraftedAndGold(int rows, int columns, int blocks, int threads);

	virtual void setDimensions(int rows, int columns);

	virtual void solveSystem(thrust::device_vector<float>& AT, thrust::device_vector<float>& y, thrust::device_vector<float>& output );
	
    virtual ~LinearSolverHandCraftedAndGold(){}

	double timeSpent()
	{
		return timeSpent_;
	}

private:
	thrust::device_vector<float> workspace_;
	thrust::device_vector<float> ATA_;
	thrust::device_vector<float> ATx_;

    thrust::host_vector<float> ATA_host_;
    thrust::host_vector<float> ATx_host_;
    thrust::host_vector<float> output_host_;

	int rows_;
	int columns_;
	int blocks_;
	int threads_;
	double timeSpent_;

    SVDSquareMatrix SVDsolver;
};

/*
class LinearSolverHandCraftedAndCula : public 	LinearSolverBaseMain
{
public: 
	LinearSolverHandCraftedAndCula(int rows, int columns, int blocks, int threads);

	virtual void setDimensions(int rows, int columns);

	virtual void solveSystem(thrust::device_vector<float>& AT, thrust::device_vector<float>& y, thrust::device_vector<float>& output );
	
	virtual ~LinearSolverHandCraftedAndCula();

	double timeSpent()
	{
		return timeSpent_;
	}

private:
	thrust::device_vector<float> workspace_;
	thrust::device_vector<float> ATA_;
	thrust::device_vector<float> ATx_;
	int rows_;
	int columns_;
	int blocks_;
	int threads_;
	double timeSpent_;
};
*/
#endif
