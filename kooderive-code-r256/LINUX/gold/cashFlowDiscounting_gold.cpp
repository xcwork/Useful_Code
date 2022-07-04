//
//
//                                                                          cashFlowDiscounting_gold.cpp
//
//
// (c) Mark Joshi 2010,2011
// This code is released under the GNU public licence version 3
#include <gold/cashFlowDiscounting_gold.h>
#include <cmath>
#include <gold/MatrixFacade.h> 


// This routine is only called once per simulation so will not be a bottleneck.
// It is therefore written for clarity rather than speed.
int findExerciseTimeIndicesFromPaymentTimeIndices(const std::vector<int>& genTimeIndex, 
                                                   const std::vector<int>& exerciseIndices,
                                                   const std::vector<int>& exerciseIndicators, // redundant info but sometimes easier to work with
                                                   std::vector<int>& stepToExerciseIndices
                                                   )
{
    int numberRatesAndSteps = genTimeIndex.size();
    stepToExerciseIndices.resize(numberRatesAndSteps);
    int numberExerciseTimes = exerciseIndices.size();

    for (int i=0; i < numberRatesAndSteps; ++i)
    {
        int j=0;
        while (j < numberExerciseTimes && exerciseIndices[j] < genTimeIndex[i])
            ++j;

        if (j == numberExerciseTimes)
            --j;
        if ( exerciseIndices[j] > genTimeIndex[i])
            --j;
        stepToExerciseIndices[i]=j;


    }

	int first=0;
	while(first <stepToExerciseIndices.size() && stepToExerciseIndices[first]<0)
		++first;
	return first;

}
