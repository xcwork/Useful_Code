//
//
//                                  MonteCarloStatistics_gold.cpp
//
//

/*

Copyright (c) 2012 Mark Joshi

released under the GPL v 3.0
*/

#include <gold/MonteCarloStatistics_gold.h>

void MonteCarloStatistics::AddDataVector(const std::vector<Realv>& dataPoints)
{
    AddData(&dataPoints[0]);
}
