//
//
//                                  EarlyExercisableMultiEquityPayoff_gold.h
//
//

// (c) Mark Joshi 2015
// This code is released under the GNU public licence version 3

#ifndef EARLYEXERCISABLEMULTIEQUITYPAYOFF_GOLD_H
#define EARLYEXERCISABLEMULTIEQUITYPAYOFF_GOLD_H
#include <vector>

#include <gold/MatrixFacade.h>

class MaxCall
{
public:
    MaxCall(int stocks, double Strike);

    double operator()(const std::vector<double>& currentStocks) const;

    int numberUnderlyings() const
    {
        return stocks_;
    }


private:
    int stocks_;
    double Strike_;

};

// stock price values are dimensioned step, stock number, path
// payoffs are dimensioned step, path
template<class T>
void GenerateMultiDEquityPayoffs(const CubeConstFacade<double>& stockPriceValues, 
                                    const T& payoff, 
                                    int pathOffsetForOutput,
                                    int stepsToSkip,
                                    int stepsToSkipForOutput,
                                    MatrixFacade<double>& payoffs)
{
      if (stockPriceValues.numberRows() != payoff.numberUnderlyings())
          GenerateError("dimension mismatch in GenerateMultiDEquityPayoffs");

     if (stockPriceValues.numberLayers() != payoffs.rows())
          GenerateError("dimension mismatch in GenerateMultiDEquityPayoffs");
 
     if (stockPriceValues.numberColumns() > payoffs.columns()+pathOffsetForOutput)
          GenerateError("dimension mismatch in GenerateMultiDEquityPayoffs: insufficient output paths in payoffs matrix");

     int numberStocks = payoff.numberUnderlyings();
     int paths= stockPriceValues.numberColumns();
     int steps = stockPriceValues.numberLayers();

     std::vector<double> stock_vec(numberStocks);

     for (int i=0; i < paths; ++i)
     {
         for (int s=stepsToSkip, outStep=stepsToSkipForOutput; s < steps; ++s,++outStep)
         {
             for (int k=0; k < numberStocks; ++k)
                 stock_vec[k] = stockPriceValues(s,k,i);
             payoffs(outStep,i+pathOffsetForOutput) = payoff(stock_vec);
         }
     }


}

#endif
