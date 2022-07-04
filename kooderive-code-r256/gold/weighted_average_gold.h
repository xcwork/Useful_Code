
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3

//                          weighted_average_gold.h

/*
weighted basket example

*/


#ifndef WEIGHTED_AVERAGE_GOLD_H
#define WEIGHTED_AVERAGE_GOLD_H
#include <gold/MatrixFacade.h>
#include <vector>

template<class T>
void  basketWeightings_gold(CubeConstFacade<T>& inputPaths, 
                            MatrixFacade<T>& outputPaths, 
                            const std::vector<T>& weights,
                                           int paths,
                                           int numberOfStocks,
                                           int numberSteps)
{

    if (inputPaths.numberColumns() != paths)
        GenerateError("Misshapen input for paths in basketWeightings_gold");
  
    if (inputPaths.numberRows() != numberOfStocks)
        GenerateError("Misshapen input for stocks in basketWeightings_gold");

 
    if (inputPaths.numberLayers() != numberSteps)
        GenerateError("Misshapen input for numberSteps in basketWeightings_gold");
    
    if (outputPaths.rows() != numberSteps)
        GenerateError("Misshapen output for paths in basketWeightings_gold");
       
    if (outputPaths.columns() != paths)
        GenerateError("Misshapen output for numberSteps in basketWeightings_gold");

    for (int p=0; p < paths; ++p)
        for (int step=0; step < numberSteps; ++step)
        {
            T total =static_cast<T>(0.0);

            for (int i=0; i < numberOfStocks; ++i)
                total += weights[i]*inputPaths(step,i,p);

            outputPaths(step,p) = total;

        }
  

}


#endif
