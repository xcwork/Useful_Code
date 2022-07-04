//                                      Regression_selector_gold.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef REGRESSION_SELECTOR_CONCRETE_GOLD_H
#define REGRESSION_SELECTOR_CONCRETE_GOLD_H
#include <gold/Regression_Selector_gold.h>
class RegressionSelector_gold_Fraction : public RegressionSelectorGold
{
public:
	
	RegressionSelector_gold_Fraction(double lowerFrac, double upperFrac, double initialSDguess, double multiplier)
		: lowerFrac_(lowerFrac), upperFrac_(upperFrac), initialSDguess_(initialSDguess),
		multiplier_(multiplier)
	{}
	    

	virtual int SelectF(int depth,
		        std::vector<float>::const_iterator start, 
				std::vector<float>::const_iterator end,
				std::vector<int>::iterator selected,
				std::vector<int>::const_iterator preSelected,
				int numberPreselected,
				float& lowerCutOff,
				float& upperCutOff);

	virtual int Select(int depth,
		        std::vector<double>::const_iterator start, 
				std::vector<double>::const_iterator end,
				std::vector<int>::iterator selected,
				std::vector<int>::const_iterator preSelected,
				int numberPreSelected,
				double& lowerCutOff,
				double& upperCutOff);



	virtual ~RegressionSelector_gold_Fraction()
	{}

private:
	double lowerFrac_; 
	double upperFrac_; 
	double initialSDguess_; 
	double multiplier_;
};
#endif
