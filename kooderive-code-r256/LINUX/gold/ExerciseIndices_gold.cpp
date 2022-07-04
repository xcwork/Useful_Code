//
//
//             ExerciseIndices_gold.cpp
//
//
// (c) Mark Joshi 2011,2014
// This code is released under the GNU public licence version 3
#include <gold/ExerciseIndices_gold.h>
#include <vector>

// generate indices of exercise times amongst rate times 
void GenerateIndices(const std::vector<int>& indicators, std::vector<int>& indices)
{
    for (size_t j=0; j < indicators.size(); ++j)
        if (indicators[j] > 0)
            indices.push_back(j);
}

void GenerateIndicators(std::vector<int>& indicators, const std::vector<int>& indices, int numberOfSteps)
{
    indicators.resize(0);
    indicators.resize(numberOfSteps,0);


    for (size_t j=0; j < indices.size(); ++j)
      indicators[j] = 1;
}
