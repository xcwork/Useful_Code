
//
//                                  MonteCarloStatistics_concrete_gold.cpp
//
//

/*

Copyright (c) 2012 Mark Joshi

released under the GPL v 3.0
*/
#include <gold/MonteCarloStatistics_concrete_gold.h>
#include <cmath>

MonteCarloStatisticsBatched::MonteCarloStatisticsBatched(int BatchSize_, 
                                                         size_t Dimension_, 
                                                         const std::vector<std::string>& Names_)
                                                         : BatchSize(BatchSize_),  Dimension(Dimension_),  Names(Names_),

                                                         CompletedBatchs(0),
                                                         SumBatchMeans(Dimension_,0.0),
                                                         SumBatchMeansSquares(Dimension_,0.0),
                                                         SumBatchVariances(Dimension_,0.0),
                                                         SumBatchVariancesSquares(Dimension_,0.0),
                                                         PathNumber(0),
                                                         SumCurrentBatch(Dimension_,0.0),
                                                         SumSqCurrentBatch(Dimension_,0.0)
{

}


void MonteCarloStatisticsBatched::AddData(const Realv* const dataPoints)
{
    for (size_t i=0; i < Dimension; ++i)
    {
        Realv x= dataPoints[i];
        SumCurrentBatch[i]+=x;
        SumSqCurrentBatch[i]+=x*x;
    }

    ++PathNumber;

    if (PathNumber<BatchSize)
        return;

    for (size_t i=0; i < Dimension; ++i)
    {
        Realv x = SumCurrentBatch[i];
        Realv y = SumSqCurrentBatch[i];
        Realv mean = x/BatchSize;
        Realv variance = y/BatchSize - mean*mean;

        SumBatchMeans[i]+=mean;
        SumBatchMeansSquares[i]+=mean*mean;
        SumBatchVariances[i]+=variance;
        SumBatchVariancesSquares[i]+=variance*variance;

        SumCurrentBatch[i]=0.0;
        SumSqCurrentBatch[i]=0.0;
        PathNumber=0;
        ++CompletedBatchs;



    }


}

size_t MonteCarloStatisticsBatched::DataDimension() const
{
    return Dimension;
}

std::vector<std::vector<Realv> > MonteCarloStatisticsBatched::GetStatistics() const
{
    std::vector<std::vector<Realv> > results(5);

    for (size_t i=0; i < 4; ++i)
        results[i].resize(Dimension);

    for (size_t j=0; j < Dimension; ++j)
    {
        Realv mean = SumBatchMeans[j]/CompletedBatchs;
        results[0][j] = mean;
        Realv meanvar = SumBatchMeansSquares[j]/CompletedBatchs -mean*mean;
        results[1][j] = sqrt(meanvar/CompletedBatchs);

        Realv meanV = SumBatchVariances[j]/CompletedBatchs;
        results[2][j] = meanV;
        Realv varvar = SumBatchVariancesSquares[j]/CompletedBatchs -meanV*meanV;
        results[3][j] = sqrt(varvar/CompletedBatchs);
    }

    results[4].resize(1);
    results[4][0] = CompletedBatchs;

    return results;
}

std::vector<std::string> MonteCarloStatisticsBatched::GetStatisticsNames() const
{
    return Names;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MonteCarloStatisticsSimple::MonteCarloStatisticsSimple(
                                                         size_t Dimension_, 
                                                         const std::vector<std::string>& Names_)
                                                         :   Dimension(Dimension_),  Names(Names_),
                                                         PathNumber(0),
                                                         SumCurrentBatch(Dimension_,0.0),
                                                         SumSqCurrentBatch(Dimension_,0.0)
{

}
MonteCarloStatisticsSimple::MonteCarloStatisticsSimple(
                                                         size_t Dimension_)
                                                         :   Dimension(Dimension_),  Names(Dimension_),
                                                         PathNumber(0),
                                                         SumCurrentBatch(Dimension_,0.0),
                                                         SumSqCurrentBatch(Dimension_,0.0)
{

}


void MonteCarloStatisticsSimple::AddData(const Realv* const dataPoints)
{
    for (size_t i=0; i < Dimension; ++i)
    {
        Realv x= dataPoints[i];
        SumCurrentBatch[i]+=x;
        SumSqCurrentBatch[i]+=x*x;
    }

    ++PathNumber;

   


}

size_t MonteCarloStatisticsSimple::DataDimension() const
{
    return Dimension;
}

std::vector<std::vector<Realv> > MonteCarloStatisticsSimple::GetStatistics() const
{
    std::vector<std::vector<Realv> > results(3);

    for (size_t i=0; i < 2; ++i)
        results[i].resize(Dimension);

    for (size_t j=0; j < Dimension; ++j)
    {
        Realv mean = SumCurrentBatch[j]/PathNumber;
        results[0][j] = mean;
        Realv variance = ( SumSqCurrentBatch[j]/(PathNumber) - mean*mean)*PathNumber/(PathNumber-1.0);
        results[1][j] =variance >0.0?  sqrt(variance/PathNumber) :0.0;
    }

    results[2].resize(1);
    results[2][0] = PathNumber;

    return results;
}

std::vector<std::string> MonteCarloStatisticsSimple::GetStatisticsNames() const
{
    return Names;

}
