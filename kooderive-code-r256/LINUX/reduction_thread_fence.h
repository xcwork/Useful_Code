// reduction_thread_fence.h
// adapted from SDK example


// released under GPL v 3
// (c) Mark Joshi 2013

#ifndef reduction_thread_fence_h
#define reduction_thread_fence_h
#include <thrust/device_vector.h>

extern "C"
void reduceSinglePass(int size, int threads, int blocks, float *idata_global, float *workspace_global, float* final_loc_global);

class ReductionThreadFence
{
public:
	ReductionThreadFence(int size, int threads=0, int blocks=0);

	void reduce(float* idata_global, float* final_loc_global);

private:
	int size_;
	int threads_;
	int blocks_;

	thrust::device_vector<float> wsp_dev_;
	float* wsp_global_;
};

#endif
