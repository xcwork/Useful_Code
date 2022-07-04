//
//
//                         RegressionSelectorConcrete.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3


#ifndef REGRESSION_SELECTOR_CONCRETE_H
#define REGRESSION_SELECTOR_CONCRETE_H
#include <RegressionSelectorBase.h>
/*
marks selected elements with a 1 
not selected with a zero
return total number selected
*/

/*
RegressionSelectorStandardDeviations 
selects on how many standard deviations from zero
*/

class RegressionSelectorStandardDeviations : public RegressionSelector
{
public:
	
	RegressionSelectorStandardDeviations(float numberSdsForCutOff);

 virtual int Select(int depth,
		        thrust::device_vector<float>::iterator& start, 
		        thrust::device_vector<float>::iterator& end,
		        thrust::device_vector<int>::iterator& selected,
				float& lowerCutOff,
				float& upperCutOff);

 
	virtual int Select_gold(int depth,
				const std::vector<float>& data,
				int DataSize,
				std::vector<int>& selected,
				float& lowerCutOff,
				float& upperCutOff);



private:
	float numberSdsForCutOff_;

};


class RegressionSelectorFraction : public RegressionSelector
{
public:
	
	RegressionSelectorFraction(float lowerFrac, float upperFrac, float initialSDguess, float multiplier);
	
	RegressionSelectorFraction(double lowerFrac, double upperFrac, double initialSDguess, double multiplier);

 virtual int Select(int depth,
		        thrust::device_vector<float>::iterator& start, 
		        thrust::device_vector<float>::iterator& end,
		        thrust::device_vector<int>::iterator& selected,
				float& lowerCutOff,
				float& upperCutOff);

 
	virtual int Select_gold(int depth,
				const std::vector<float>& data,
				int DataSize,
				std::vector<int>& selected,
				float& lowerCutOff,
				float& upperCutOff);

	double timeInHere()
	{
		return timeSpent_;
	}

private:
	float lowerFrac_;
	float upperFrac_;
	float numberSdsForCutOff_;
	float multiplier_;
	double timeSpent_;

};


#endif


