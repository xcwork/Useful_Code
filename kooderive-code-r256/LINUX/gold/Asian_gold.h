// (c) Mark Joshi 2009,2014
// This code is released under the GNU public licence version 3



#ifndef ASIAN_GOLD_H
#define ASIAN_GOLD_H
#include <gold/pragmas.h>
#include <vector>

template<class D>
void AsianCallGPU_gold (const D* const input_normals,
                                                                                                 int normalsDimensions, // must be greater than or equal to stepsPerPath
                                                                                                 int totalPaths, 
                                                                                                 int stepsPerPath,
                                                                                                 const std::vector<D>& logDrifts_vec, 
                                                                                                 const std::vector<D>& logSds_vec, 
                                                                                                 D logSpot0,
                                                                                                 D df,
                                                                                                 D strikeArithmetic,
                                                                                                 std::vector<D>& outputDataArithmetic,
                                                                                                 D strikeGeometric,
                                                                                                 std::vector<D>&outputDataGeometric)
{
    
    if  (logDrifts_vec.size() != stepsPerPath || logSds_vec.size() != stepsPerPath)
        throw("Size mismatch");

    outputDataArithmetic.resize(totalPaths);
    outputDataGeometric.resize(totalPaths);

    for (int i=0; i < totalPaths; ++i)
    {  
            D logSpot = logSpot0;
            D runningSum =0.0;
            D runningSumLog =0.0;

            const D* const normalPtr = input_normals+i;

            for (int j=0; j < stepsPerPath; ++j)
            {
                D variate =  normalPtr[j*totalPaths];
                logSpot += logDrifts_vec[j] +variate*logSds_vec[j];
                D spot = exp(logSpot);
                 runningSum += spot/stepsPerPath;
                 runningSumLog += logSpot/stepsPerPath;
            }

            D average = runningSum;
            D payOff =  average- strikeArithmetic;
            payOff = payOff   > 0.0f ? payOff : 0.0f;
            outputDataArithmetic[i] = payOff*df;

            D payOffGeom = exp(runningSumLog) - strikeGeometric;
            payOffGeom = payOffGeom   > 0.0f ? payOffGeom : 0.0f;
            outputDataGeometric[i] = payOffGeom*df;
    }
    
}

#endif
