//
//
//                                  Compute_Statistics_gold.h
//
//
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3

#ifndef COMPUTE_STATISTICS_GOLD_H
#define COMPUTE_STATISTICS_GOLD_H

#include <gold/math/typedefs_math_gold.h>
#include <numeric>
template<class it>
void MonteCarloSimpleStatistics(it start, it end, Realv& mean, Realv& variance, Realv& standardError)
{

    Realv sum = std::accumulate(start,end,0.0);
    Realv sumsq = std::inner_product(start,end,start,0.0);
    int N = end-start;
    mean = sum/N;
    variance = sumsq/N - mean*mean;
    standardError = sqrt(variance/N);

}

#endif
