//
//
//                               RegressionSelectorConcrete.cu
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3


#include <RegressionSelectorConcrete.h>
#include <thrust/count.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <smallFunctions.h>
#include <gold/Timers.h>
	


RegressionSelectorStandardDeviations::RegressionSelectorStandardDeviations(float numberSdsForCutOff) : numberSdsForCutOff_(numberSdsForCutOff)
{}


int RegressionSelectorStandardDeviations::Select(int depth,
												 thrust::device_vector<float>::iterator& start, 
												 thrust::device_vector<float>::iterator& end,
												 thrust::device_vector<int>::iterator& selected,
												 float& lowerCutOff,
												 float& upperCutOff)
{
	int dataSize = end - start;
	float sum = thrust::reduce(start,end);
	float sumsq = thrust::transform_reduce(start,end,squareof(),0.0f,thrust::plus<float>());
	float mean = sum/dataSize;
	float sd = sqrt( sumsq/(dataSize) - mean*mean);

	upperCutOff =  numberSdsForCutOff_*sd;
	lowerCutOff = - upperCutOff;

	inInterval selector(lowerCutOff,upperCutOff);

	thrust::transform(start,end,selected,selector);

	return thrust::count(selected,selected+dataSize,1);

}


int  RegressionSelectorStandardDeviations::Select_gold(int depth,
													   const std::vector<float>& data,
													   int dataSize,
													   std::vector<int>& selected,
													   float& lowerCutOff,
													   float& upperCutOff)
{
	double sum = 0.0;

	double sumsq =0.0;

	for (int i=0; i< dataSize; ++i)
	{
		double x = data[i];
		sum += x;
		sumsq+= x*x;

	}

	double mean = sum/dataSize;
	double variance =  sumsq/dataSize - mean*mean;

	double sd = sqrt(variance);

	upperCutOff =  static_cast<float>(numberSdsForCutOff_*sd);
	lowerCutOff = - upperCutOff;

	int count =0;
	for (int i=0; i < dataSize;++i)
		if ( lowerCutOff < data[i] && data[i] < upperCutOff)
		{
			selected[i] = 1;
			++count;
		}
		else
			selected[i]=0;


	return count;


}

RegressionSelectorFraction::RegressionSelectorFraction(float lowerFrac, float upperFrac, float initialSDguess, float multiplier)
: lowerFrac_(lowerFrac), upperFrac_(upperFrac), numberSdsForCutOff_(initialSDguess), multiplier_(multiplier),timeSpent_(0.0)
{}

RegressionSelectorFraction::RegressionSelectorFraction(double lowerFrac, double upperFrac, double initialSDguess, double multiplier)
: lowerFrac_(static_cast<float>(lowerFrac)), upperFrac_(static_cast<float>(upperFrac)), numberSdsForCutOff_(static_cast<float>(initialSDguess)), multiplier_(static_cast<float>(multiplier))
,timeSpent_(0.0)
{}


int RegressionSelectorFraction::Select(int depth,
									   thrust::device_vector<float>::iterator& start, 
									   thrust::device_vector<float>::iterator& end,
									   thrust::device_vector<int>::iterator& selected,
									   float& lowerCutOff,
									   float& upperCutOff)
{
	Timer theTimer;
    float accuracy = 1E-6f; // if bracketed to within this level stop

	int dataSize = end - start;
	float sum = thrust::reduce(start,end);
	float sumsq = thrust::transform_reduce(start,end,squareof(),0.0f,thrust::plus<float>());
	float mean = sum/dataSize;
	float sd = sqrt( sumsq/(dataSize) - mean*mean);

	upperCutOff =  numberSdsForCutOff_*sd;
	lowerCutOff = - upperCutOff;

	inInterval selector(lowerCutOff,upperCutOff);

	thrust::transform(start,end,selected,selector);

	int numberSelected =  thrust::count(selected,selected+dataSize,1);

	int lowTarget = static_cast<int>(dataSize*lowerFrac_);
	int highTarget = static_cast<int>(dataSize*upperFrac_);


	if (numberSelected > lowTarget && numberSelected < highTarget)
		return numberSelected;

	float lowGuess, highGuess,guess;

	guess= numberSdsForCutOff_*sd;

	if (numberSelected < lowTarget)
	{	
		while ( numberSelected < lowTarget)
		{
			lowGuess =guess;
			guess /= multiplier_;
			inInterval selector(-guess,guess);

			thrust::transform(start,end,selected,selector);

			numberSelected =  thrust::count(selected,selected+dataSize,1);


		}
		highGuess = guess;

	}
	else
	{	
		while ( numberSelected > highTarget)
		{
			highGuess =guess;
			guess *= multiplier_;
			inInterval selector(-guess,guess);

			thrust::transform(start,end,selected,selector);

			numberSelected =  thrust::count(selected,selected+dataSize,1);


		}
		lowGuess = guess;

	}

	if (numberSelected > lowTarget && numberSelected < highTarget)
	{
		upperCutOff = guess;
		lowerCutOff= - guess;
		return numberSelected;
	}

	guess = 0.5f*(lowGuess + highGuess);
	inInterval selector2(-guess,guess);

	thrust::transform(start,end,selected,selector2);

	numberSelected =  thrust::count(selected,selected+dataSize,1);


	while ( (numberSelected < lowTarget || numberSelected > highTarget) && (highGuess - lowGuess > accuracy))
	{
		if (numberSelected < lowTarget)
			lowGuess = guess;
		else
			highGuess = guess;

		guess = 0.5f*(lowGuess + highGuess);
		inInterval selector(-guess,guess);

		thrust::transform(start,end,selected,selector);

		numberSelected =  thrust::count(selected,selected+dataSize,1);

	}


	upperCutOff = guess;
	lowerCutOff= - guess;

	timeSpent_=theTimer.timePassed();

	return numberSelected;


}

namespace
{
	int SelectAndCount(float lowerCutOff, float upperCutOff, int dataSize,  const std::vector<float>& data,  std::vector<int>& selected)
	{
		int count=0;
		for (int i=0; i < dataSize;++i)
			if ( lowerCutOff < data[i] && data[i] < upperCutOff)
			{
				selected[i] = 1;
				++count;
			}
			else
				selected[i]=0;

		return count;


	}

}
int  RegressionSelectorFraction::Select_gold(int depth,
											 const std::vector<float>& data,
											 int dataSize,
											 std::vector<int>& selected,
											 float& lowerCutOff,
											 float& upperCutOff)
{
	double sum = 0.0;

	double sumsq =0.0;

	for (int i=0; i< dataSize; ++i)
	{
		double x = data[i];
		sum += x;
		sumsq+= x*x;

	}

	double mean = sum/dataSize;
	double variance =  sumsq/dataSize - mean*mean;

	double sd = sqrt(variance);

	upperCutOff =  static_cast<float>(numberSdsForCutOff_*sd);
	lowerCutOff = - upperCutOff;

	int count = SelectAndCount( lowerCutOff, upperCutOff, dataSize, data,  selected);


	int lowTarget = static_cast<int>(dataSize*lowerFrac_);
	int highTarget = static_cast<int>(dataSize*upperFrac_);


	if (count > lowTarget && count < highTarget)
		return count;

	float lowGuess, highGuess,guess;

	guess= static_cast<float>(numberSdsForCutOff_*sd);

	if (count < lowTarget)
	{	
		while ( count < lowTarget)
		{
			lowGuess =guess;
			guess /= multiplier_;
		
			count =  SelectAndCount( -guess, guess, dataSize, data,  selected);


		}
		highGuess = guess;

	}
	else
	{	
		while ( count > highTarget)
		{
			highGuess =guess;
			guess *= multiplier_;
				
			count =  SelectAndCount( -guess, guess, dataSize, data,  selected);


		}
		lowGuess = guess;

	}

	if (count > lowTarget && count < highTarget)
	{
		upperCutOff = guess;
		lowerCutOff= - guess;
		return count;
	}

	guess = 0.5f*(lowGuess + highGuess);
	count =  SelectAndCount( -guess, guess, dataSize, data,  selected);



	while ( count < lowTarget || count > highTarget)
	{
		if (count < lowTarget)
			lowGuess = guess;
		else
			highGuess = guess;

		guess = 0.5f*(lowGuess + highGuess);
		count =  SelectAndCount( -guess, guess, dataSize, data,  selected);


	}


	upperCutOff = guess;
	lowerCutOff= - guess;
	return count;
}



