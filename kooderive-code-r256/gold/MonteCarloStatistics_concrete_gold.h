
//
//                                  MonteCarloStatistics_concrete_gold.h
//
//

/*

Copyright (c) 2012 Mark Joshi

released under the GPL v 3.0
*/

#ifndef MONTECARLOSTATISTICS_CONCRETE_H
#define MONTECARLOSTATISTICS_CONCRETE_H
#include <gold/MonteCarloStatistics_gold.h>
#include <gold/math/typedefs_math_gold.h>
#include <vector>
#include <string>

class MonteCarloStatisticsBatched : public MonteCarloStatistics
{

public:
    MonteCarloStatisticsBatched(int BatchSize_, size_t Dimension_, const std::vector<std::string>& Names_);

    virtual void AddData(const Realv* const dataPoints);
    virtual size_t DataDimension() const;

    virtual std::vector<std::vector<Realv> > GetStatistics() const;
    virtual std::vector<std::string> GetStatisticsNames() const;

  

private:
    int BatchSize;
    size_t Dimension;
    std::vector<std::string> Names;

    // varying members
    int CompletedBatchs;
    std::vector<Realv> SumBatchMeans;
    std::vector<Realv> SumBatchMeansSquares;

    std::vector<Realv> SumBatchVariances;
    std::vector<Realv> SumBatchVariancesSquares;

    int PathNumber;
    std::vector<Realv> SumCurrentBatch;
    std::vector<Realv> SumSqCurrentBatch;
   


};

class MonteCarloStatisticsSimple : public MonteCarloStatistics
{

public:
    MonteCarloStatisticsSimple( size_t Dimension_, const std::vector<std::string>& Names_);
  MonteCarloStatisticsSimple( size_t Dimension_);

    virtual void AddData(const Realv* const dataPoints);
    virtual size_t DataDimension() const;

    virtual std::vector<std::vector<Realv> > GetStatistics() const;
    virtual std::vector<std::string> GetStatisticsNames() const;

  

private:
    size_t Dimension;
    std::vector<std::string> Names;

   
   //varying members
    int PathNumber;
    std::vector<Realv> SumCurrentBatch;
    std::vector<Realv> SumSqCurrentBatch;
   


};

#endif
