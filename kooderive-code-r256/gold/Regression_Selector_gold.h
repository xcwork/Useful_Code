//
//
//                                      Regression_selector_gold.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef REGRESSION_SELECTOR_GOLD_H
#define REGRESSION_SELECTOR_GOLD_H

#include <vector>
/*
marks selected elements with a 1 
not selected with a zero
return total number selected

abstract base class
*/

class RegressionSelectorGold
{
public:
	
	RegressionSelectorGold()
	{}


	virtual int SelectF(int depth,
		        std::vector<float>::const_iterator start, 
				std::vector<float>::const_iterator end,
				std::vector<int>::iterator selected,
				std::vector<int>::const_iterator preSelected,
				int numberPreSelected,
				float& lowerCutOff,
				float& upperCutOff)=0;

	virtual int Select(int depth,
		        std::vector<double>::const_iterator start, 
				std::vector<double>::const_iterator end,
				std::vector<int>::iterator selected,
				std::vector<int>::const_iterator preSelected,
				int numberPreSelected,
				double& lowerCutOff,
				double& upperCutOff)=0;



	virtual ~RegressionSelectorGold()
	{}


};

#endif
