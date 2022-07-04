//
//
//             ExerciseIndices_gold.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

#ifndef EXERCISE_INDICES_GOLD_H
#define EXERCISE_INDICES_GOLD_H

#include <vector>

// generate indices of exercise times amongst rate times from indicators

void GenerateIndices(const std::vector<int>& indicators, std::vector<int>& indices);


// generator indicators from indices
void GenerateIndicators(std::vector<int>& indicators, const std::vector<int>& indices, int numberOfSteps);

#endif
