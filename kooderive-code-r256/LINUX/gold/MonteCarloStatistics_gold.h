//
//
//                                  MonteCarloStatistics_gold.h
//
//

/*

Copyright (c) 2012 Mark Joshi

released under the GPL v 3.0
*/

#ifndef MONTECARLOSTATISTICS_H
#define MONTECARLOSTATISTICS_H
#include <gold/math/typedefs_math_gold.h>
#include <vector>
#include <string>

class MonteCarloStatistics
{

public:
    MonteCarloStatistics(){}
    virtual ~ MonteCarloStatistics(){}

    virtual void AddData(const Realv* const dataPoints)=0;
    virtual size_t DataDimension() const=0;

    virtual std::vector<std::vector<Realv> > GetStatistics() const=0;
    virtual std::vector<std::string> GetStatisticsNames() const=0;

    void AddDataVector(const std::vector<Realv>& dataPoints);

private:


};

#endif
