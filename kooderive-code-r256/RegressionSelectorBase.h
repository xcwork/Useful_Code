//
//
//                                      RegressionSelectorBase.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef REGRESSION_SELECTOR_H
#define REGRESSION_SELECTOR_H
#include <thrust/device_vector.h>
#include <vector>
/*
marks selected elements with a 1 
not selected with a zero
return total number selected

abstract base class
*/

class RegressionSelector
{
public:
	
	RegressionSelector()
	{}


	virtual int Select(int depth,
		        thrust::device_vector<float>::iterator& start, 
				thrust::device_vector<float>::iterator& end,
				thrust::device_vector<int>::iterator& selected,
				float& lowerCutOff,
				float& upperCutOff)=0;


	virtual int Select_gold(int depth,
				const std::vector<float>& data,
				int DataSize,
				std::vector<int>& selected,
				float& lowerCutOff,
				float& upperCutOff)=0;

	// not virtual, calls the other two
	int Select_test_mode(int depth,
		        thrust::device_vector<float>::iterator& start, 
				thrust::device_vector<float>::iterator& end,
				thrust::device_vector<int>::iterator& selected,
				float& lowerCutOff,
				float& upperCutOff,
				bool & testPassed);

	virtual ~RegressionSelector()
	{}


};

#endif
