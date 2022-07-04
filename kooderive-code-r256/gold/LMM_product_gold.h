//
//
//                                  LMM_product_gold.h
//
//

// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3

#ifndef LMM_PRODUCT_GOLD_H
#define LMM_PRODUCT_GOLD_H
#include <vector>
#include <gold/MatrixFacade.h> 

/*!
Gold products for cash-flow generation using rates already generated. These will be passed into 

cashFlowGenerator_gold

as template arguments. 


constructor 

    T(const std::vector& auxData, int numberSteps);

    The vector auxData will contain things like accruals, target coupon spreads. 


 Note the design here is that the product does not know anything about what rates it depends on or cash-flow payment times. 
 The client routines using it decide these things elsewhere. This enables us to use the same product for CMS rates as for LIBOR rates. 

Two main methods: 

1: void newPath() -- puts product at the start of a new paths
2:  bool product.getCashFlows( D& flow1,   // for passing first flow back
                                              D& flow2,   // for passing second flow back
                                              D rate1, // first rate that determines the flow eg a LIBOR or CMS rate 
                                              D rate2,  // second rate that determines the flow eg a CMS rate 
                                              D rate3,  // third rate that determines the flow eg another CMS rate  (or just ignore) 
                                              CubeConstFacade<D>& forwardsCube, // all the underlying forward rates, steps \times rates \times paths 
                                              CubeConstFacade<D>& discountsCube,// all the underlying discount ratios, steps \times 1+rates \times paths 
                                              CubeConstFacade<D>& cotermSwapRatesCube, // all the underlying  coterminal swap rates  , steps \times rates \times paths 
                                              int p // current path
                                               );

the return value indicates whether a path is finished. eg the product has cancelled or been triggered. 

The last few arguments will generally not be needed but since the data will generally already have been computed, always providing it does little harm and 
avoids recomputations when it is needed. The path is provided to enable extraction from the cubes. 

Many products will only use one or two rates, in this case the third passed in can be ignored.

The implicit assumptions of this product are 
1) at each step, cash-flows can only be generated at two times corresponding to flow1 and flow2. 
2) the product is stepped through the rate times so number steps = number rates. 
*/

/*!
product_TARN_gold 

Does a TARN paying an inverse floating coupon and receive floating. 

*/
template<class D>
class product_TARN_gold
{
public:
    product_TARN_gold(const std::vector<D>& auxData, int numberSteps);
    
    void newPath(); 
    
    bool getCashFlows( D& flow1,   // for passing first flow back
                                              D& flow2,   // for passing second flow back
                                              D rate1, // first rate that determines the flow eg a LIBOR or CMS rate 
                                              D rate2,  // second rate that determines the flow eg a CMS rate 
                                              D rate3,  // third rate that determines the flow eg another CMS rate  (or just ignore) 
                                              CubeConstFacade<D>& forwardsCube, // all the underlying forward rates, steps \times rates \times paths 
                                              CubeConstFacade<D>& discountsCube,// all the underlying discount ratios, steps \times 1+rates \times paths 
                                              CubeConstFacade<D>& cotermSwapRatesCube, // all the underlying  coterminal swap rates  , steps \times rates \times paths 
                                              int p // current path
                                               );
private:
    int numberSteps_;

    D  totalCoupon_;
    D strike_;
    D multiplier_;

    std::vector<D> accruals1_;
    std::vector<D> accruals2_;


    // changing value
    D couponPaid_;
    int currentStep_;
    bool done_;
};
template<class D>
product_TARN_gold<D>::product_TARN_gold(const std::vector<D>& auxData, int numberSteps) : numberSteps_(numberSteps), 
                                                                                            accruals1_(numberSteps), 
                                                                                           accruals2_(numberSteps)
{
    if (static_cast<int>(auxData.size()) < 2*numberSteps+3)
        throw("Insufficient aux data given to product_TARN_gold");

    totalCoupon_= auxData[0];
    strike_= auxData[1];
    multiplier_= auxData[2];

    for (int i=0; i < numberSteps; ++i)
    {
          accruals1_[i]= auxData[3+i];
          accruals2_[i]= auxData[3+numberSteps+i];
    }

    done_ =true; // product should not be used without newPath being called first. 

}
    
template<class D>
void product_TARN_gold<D>::newPath()
{
    couponPaid_=0;
    currentStep_=0;
    done_=false;
}
template<class D>
bool product_TARN_gold<D>::getCashFlows( D& flow1,   // for passing first flow back
                                              D& flow2,   // for passing second flow back
                                              D rate1, // first rate that determines the flow eg a LIBOR or CMS rate 
                                              D rate2,  // second rate that determines the flow eg a CMS rate 
                                              D rate3,  // third rate that determines the flow eg another CMS rate  (or just ignore) 
                                              CubeConstFacade<D>& forwardsCube, // all the underlying forward rates, steps \times rates \times paths 
                                              CubeConstFacade<D>& discountsCube,// all the underlying discount ratios, steps \times 1+rates \times paths 
                                              CubeConstFacade<D>& cotermSwapRatesCube, // all the underlying  coterminal swap rates  , steps \times rates \times paths 
                                              int p // current path
                                               )
{
        if (done_)
        {
            flow1=0.0;
            flow2=0.0;
            return true;
        }

        
        D accrual1 = accruals1_[currentStep_];
        D accrual2 = accruals2_[currentStep_];

        flow2 = rate2*accrual2;

        if (currentStep_ +1 == numberSteps_)
        {
                D remainingCoupon = totalCoupon_ - couponPaid_;
                flow1 = - remainingCoupon;

                done_ = true;
                return true;
        } 

        D thisRate = std::max<D>(strike_ - multiplier_*rate1,0.0f);

        D thisCoupon = thisRate*accrual1;

        if (thisCoupon+ couponPaid_ >= totalCoupon_)
        {
                D remainingCoupon = totalCoupon_ - couponPaid_;
                flow1 = - remainingCoupon;

                done_ = true;
                return true;
        }

        couponPaid_+= thisCoupon;
        
        flow1 = - thisCoupon;
        return false;
}



#endif
