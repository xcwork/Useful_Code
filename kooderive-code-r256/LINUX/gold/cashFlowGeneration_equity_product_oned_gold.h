//
//
//                                  cashFlowGeneration_equity_product_oned_gold.h
//
//

// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3

#ifndef CASH_FLOW_GEN_EQ_PROD_ONE_GOLD_H
#define CASH_FLOW_GEN_EQ_PROD_ONE_GOLD_H

#include <vector>
#include <gold/MatrixFacade.h>
#include <vector>
#include<gold/cashFlowGeneration_equity_oned_gold.h>

template<class D>
void RatchetPayoffs_gold(  std::vector<D>& genFlows,
                      D strike,
                      int numberSteps,
                      int paths,
                        const MatrixConstFacade<D>& stockpaths);


template<class D>
class RatchetPutGold
{
public:

    RatchetPutGold(const std::vector<D>& auxData, int numberSteps)
      : numberSteps_(numberSteps), strike_(auxData[0])
      {}

      D getPayoff(const MatrixConstFacade<D>& stockpaths, int pathNumber) const;

private:

    int numberSteps_;
    D strike_;


};

template<class D>
D RatchetPutGold<D>::getPayoff(const MatrixConstFacade<D>& stockpaths, int p) const
{

  
        D strike = strike_;
        for (int i=0; i < numberSteps_-1; ++i)
            strike=std::max(strike,stockpaths(i,p));

        D payoff = std::max<D>(strike- stockpaths(numberSteps_-1,p),0.0);
    
        return payoff;
}

template<class D>
void RatchetPayoffs_gold(  std::vector<D>& genFlows,
                      D strike,
                      int numberSteps,
                      int paths,
                        const MatrixConstFacade<D>& stockpaths)
{

    std::vector<D> auxData(1);
    auxData[0] = strike;
    
    cashFlowGenerator_equity_oned_gold<RatchetPutGold<D>,D>(
                                         genFlows ,
                             auxData, 
                            paths, 
                            numberSteps,                       
                            stockpaths);  

}

#endif
