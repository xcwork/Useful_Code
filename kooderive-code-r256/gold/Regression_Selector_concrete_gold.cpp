//                                      Regression_selector_gold.cpp
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3
#include <gold/Regression_Selector_concrete_gold.h>
#include<vector>
#include <numeric>
#include <cmath>
namespace
{
	double tolerance = 1E-6;
	double toleranceF = 1e-5;

	template<class S, class T>
	int DoSelection(T start, 
		T end,
		std::vector<int>::iterator selected,
		std::vector<int>::const_iterator preSelected,
		S lowCut,
		S highCut
		)
	{
		
		int count =0;
		for (T it=start; it != end; ++it, ++selected,++preSelected)
		{
			if (*it >= lowCut && *it <= highCut && *preSelected)
			{
				*selected = true;
				++count;
			}
			else
				*selected = false;
		}
		return count;

	}

	template<class T>
	int CountSelection(T start, 
		T end,
		std::vector<int>::const_iterator preSelected,
		double lowCut,
		double highCut
		)
	{
		int count =0;
		for (T  it=start; it != end; ++it,++preSelected)
		{
			if (*it >= lowCut && *it <= highCut && *preSelected)
				++count;
	
		}
		return count;

	}

	template<class S, class T>
	int TheSelect(int depth,
		T start, 
		T end,
		std::vector<int>::iterator selected,
		std::vector<int>::const_iterator preSelected,
		int numberPreSelected,
		S& lowerCutOff,
		S& upperCutOff,
		double lowerFrac,
		double upperFrac,
		double numberSdsForCutOff,
		double multiplier,
		double tolerance)
	{
		int dataSize = numberPreSelected;

		int lowerTarget = static_cast<int>(dataSize*lowerFrac);
		int upperTarget = static_cast<int>(dataSize*upperFrac);

		if (numberPreSelected > lowerTarget && numberPreSelected < upperTarget)
		{
				lowerCutOff =- static_cast<S>(10000);
			upperCutOff = static_cast<S>(10000);
			return DoSelection( start, 
		                        end,
		                        selected,preSelected, lowerCutOff,upperCutOff);

		}

		double sum=0.0;
		double sumsq=0.0;
	
		{
			std::vector<int>::const_iterator it2 = preSelected;

		for (T it = start; it !=end; ++it, ++it2)
			if (*it2)
			{
				double x= *it;
				sum += x;
				sumsq +=x*x;
			}
		}

		double mean = sum/numberPreSelected;
		

		double sd = sqrt(sumsq/numberPreSelected - mean*mean);

		double guess = sd*numberSdsForCutOff;

		int count = CountSelection(start,end,preSelected,-guess,guess);
		
		while( count < lowerTarget)
		{
			guess/=multiplier;
			count = CountSelection(start,end,preSelected,-guess,guess);
		}

		if (count <upperTarget)
		{
			lowerCutOff =- static_cast<S>(guess);
			upperCutOff = static_cast<S>(guess);
			return DoSelection( start, 
		                        end,
		                        selected,preSelected, lowerCutOff,upperCutOff);
		}

		// count is bigger than upper target

		double lowGuess = guess*multiplier;
		count = CountSelection(start,end,preSelected,-lowGuess,lowGuess);
		
		while( count > upperTarget)
		{
			lowGuess*=multiplier;
			count = CountSelection(start,end,preSelected,-lowGuess,lowGuess);
		}


		if (count >lowerTarget)
		{
				upperCutOff = static_cast<S>(lowGuess);
		
			   lowerCutOff =- upperCutOff;
			return DoSelection( start, 
		                        end,
		                        selected,preSelected, lowerCutOff,upperCutOff);
		}

		// we now have guess bigger than upper target, and low guess less than lower target

		double highGuess = guess;

		guess = 0.5*(lowGuess+highGuess);
		count = CountSelection(start,end,preSelected,-guess,guess);

		if (count >= lowerTarget && count <= upperTarget)
		{
			upperCutOff = static_cast<S>(guess);
			lowerCutOff =-  static_cast<S>(guess);
		
			return DoSelection( start, 
		                        end,
		                        selected,preSelected, lowerCutOff,upperCutOff);

		}

		while ((count < lowerTarget || count > upperTarget) && (highGuess - lowGuess) > tolerance)
		{
			if (count < lowerTarget)
			{
				lowGuess = guess;
			}
			else
				highGuess = guess;

			guess = 0.5*(lowGuess+highGuess);
			count = CountSelection(start,end,preSelected,-guess,guess);

		}
		upperCutOff = static_cast<S>(guess);
		lowerCutOff =- upperCutOff;
		
		return DoSelection( start, 
		                        end,
		                        selected,preSelected, lowerCutOff,upperCutOff);

	}



}

 int RegressionSelector_gold_Fraction::Select(int depth,
		        std::vector<double>::const_iterator start, 
				std::vector<double>::const_iterator end,
				std::vector<int>::iterator selected,
				std::vector<int>::const_iterator preSelected,
				int numberPreSelected,
				double& lowerCutOff,
				double& upperCutOff)
{
	return TheSelect( depth,
		start, 
		 end,
		 selected,
		 preSelected,
		 numberPreSelected,
		lowerCutOff,
		 upperCutOff,
		 lowerFrac_,
		 upperFrac_,
		 initialSDguess_,
		 multiplier_,
		 tolerance);

}
 
 int RegressionSelector_gold_Fraction::SelectF(int depth,
		        std::vector<float>::const_iterator start, 
				std::vector<float>::const_iterator end,
				std::vector<int>::iterator selected,
				std::vector<int>::const_iterator preSelected,
				int numberPreSelected,
				float& lowerCutOff,
				float& upperCutOff)
{
	return TheSelect( depth,
		start, 
		 end,
		 selected,
		 preSelected,
		 numberPreSelected,
		lowerCutOff,
		 upperCutOff,
		 lowerFrac_,
		 upperFrac_,
		 initialSDguess_,
		 multiplier_,
		 toleranceF);

}


