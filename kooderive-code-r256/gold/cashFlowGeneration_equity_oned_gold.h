//
//
//                                  cashFlowGeneration_equity_oned_gold.h
//
//

// (c) Mark Joshi 2012,2014
// This code is released under the GNU public licence version 3

#ifndef CASH_FLOW_GENERATION_EQUITY_ONED_GOLD_H
#define CASH_FLOW_GENERATION_EQUITY_ONED_GOLD_H

#include <vector>
#include <gold/MatrixFacade.h> 

template<class T,class D>
void cashFlowGenerator_equity_oned_gold(
                                        std::vector<D>& genFlows ,
                            const std::vector<D>& auxData, 
                            int paths, 
                            int numberSteps,                       
                            const MatrixConstFacade<D>& stockpaths
)   
{    
//    int numberRates = numberSteps;

    T product(auxData, numberSteps);

   

    
    for (int p=0; p < paths; ++p)
    { 
       genFlows[p] =  product.getPayoff(stockpaths,p);
    }
}

#endif
