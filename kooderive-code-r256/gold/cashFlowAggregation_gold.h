//
//
//                                              CashFlowAggregation_gold.h
//
//
// (c) Mark Joshi 2012,2014
// This code is released under the GNU public licence version 3





#include <vector>
#include <cmath>
#include <gold/MatrixFacade.h> 

template<class T>
void AggregateFlows_gold(MatrixFacade<T>& aggregatedFlows, // for output, aggregrates are added to existing data
                         MatrixConstFacade<T>& inputFlows,
                         const std::vector<int>& precedingIndex, 
                         int batchPaths,
                         int offsetForOutput)
{
	int start =0;
	while (start < precedingIndex.size() && precedingIndex[start] <0)
		++start;

    for (int p=0; p < batchPaths; ++p)
        for (int i=start; i< precedingIndex.size(); ++i)
        {
            int j = precedingIndex[i];
	        aggregatedFlows(j,p+offsetForOutput)+= inputFlows(i,p);

        }       
}
                         
                         