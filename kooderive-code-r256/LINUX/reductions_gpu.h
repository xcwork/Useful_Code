//
//
//								Reductions.h
//
//

#ifndef REDUCTIONS_H
#define REDUCTIONS_H
const float* PartialReduce_using8s_gpu(const float* data,
								 float* workspace, // at least size points
							int points,
							int threads,
							int stopPoint=1);

const float* PartialReduce_using16s_gpu(const float* data,
								 float* workspace, // at least size points
							int points,
							int threads);
#endif
