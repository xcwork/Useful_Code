

//
//                                  cashFlowGeneration_product_gpu.h
//
//

// (c) Mark Joshi 2010,2011,2013
// This code is released under the GNU public licence version 3


#ifndef CASH_FLOW_PRODUCT_GENERATION_H
#define CASH_FLOW_PRODUCT_GENERATION_H
#include <cashFlowGeneration_gpu.h>

// product specific data is passed in via the texture tex_aux_data
// pays (K-mf)_{+}tau  
// data total coupon, K, m, fixed rate accruals, floating rate accruals 
class TARN
{

public:

__device__ TARN(int offset, int numberSteps)
{
    numberSteps_ = numberSteps;
    totalCoupon_ = tex1Dfetch(tex_aux_data,offset); // the total coupon during the product's fixed life time at offset
    offset_ = offset;
    ++offset_;
    strike_ = tex1Dfetch(tex_aux_data,offset_); // the strike, K,  of the inverse floater coupons at offset plus 1
    ++offset_;
    multiplier_ = tex1Dfetch(tex_aux_data,offset_);// the rate multiplier, m,  of the inverse floater at offset plus 1
    ++offset_;
    newPath();
}

__device__ void newPath()
{
    index_=0;
    couponPaid_=0.0f;
    done_ = false;
}

__device__ int getIndex()
{
    return index_;
}

__device__ bool getCashFlows(float& flow1, float& flow2, float rate1, float rate2, float rate3, 
							 const float* __restrict__ allForwardRates, 
							 const float* __restrict__ allDiscountRatios, 
							 int paths )
{
   
    
    if (done_)
        return done_;
        
    float accrual2 = tex1Dfetch(tex_aux_data,offset_+index_+numberSteps_);
    flow2 = accrual2*rate2;
  
    if (index_ +1 == numberSteps_)
    {
        flow1 = couponPaid_-totalCoupon_ ;
        done_ = true;
        return done_;  
    }
        
    float accrual1 = tex1Dfetch(tex_aux_data,offset_+index_);
     
    float uncappedFlow = (strike_-multiplier_*rate1)*accrual1;

    if (uncappedFlow <0.0f)
        uncappedFlow =0.0f;
    
    float newTotalCoupon = couponPaid_ +uncappedFlow;
    if (newTotalCoupon >= totalCoupon_)
    {
        done_ =true;
        flow1 =  couponPaid_-totalCoupon_ ;
        return done_;
    }

    flow1 = - uncappedFlow;
    couponPaid_ = newTotalCoupon;
    
    ++index_;
    
    return done_;

}


private:
    int offset_;
    int numberSteps_;
    int index_;
    
    float totalCoupon_;
    float strike_;
    float multiplier_;
    
    float couponPaid_;
    bool done_;

};



extern "C"
void cashFlowGeneratorCallerTARN(float* genFlows1, 
                                 float* genFlows2, 
                                 float* aux_data, 
                                 int paths, 
                                 int numberSteps,
                                 float* rates1, 
                                 float* rates2, 
                                 float* rates3, 
                                 float* forwards, 
                                 float* discRatios);

/*
Swaps rate1 against a fixed coupon.

rate2 and rate3 are ignored. 
*/

class Swap
{

public:

__device__ Swap(int offset, int numberSteps)
{
    numberSteps_ = numberSteps;
    offset_ = offset;

    strike_ = tex1Dfetch(tex_aux_data,offset_);
    ++offset_;
    payReceive_ = tex1Dfetch(tex_aux_data,offset_);
    ++offset_;
 
    newPath();
}

__device__ void newPath()
{
    index_=0;
    done_ = false;
}

__device__ int getIndex()
{
    return index_;
}

__device__ bool getCashFlows(float& flow1, float& flow2, float rate1, float rate2, float rate3,
							 const float* __restrict__ allForwardRates, const float* __restrict__ allDiscountRatios,  int paths )
{
   

    float accrual2 = tex1Dfetch(tex_aux_data,offset_+index_+numberSteps_);
    flow2 = accrual2*rate1*payReceive_;
  
   // this bit could be precomputed if you really wanted to 
    float accrual1 = tex1Dfetch(tex_aux_data,offset_+index_);
    flow1 = -strike_*accrual1*payReceive_;
      
    ++index_;

    done_ = (numberSteps_ == index_);
    
    return done_;

}


private:
    int offset_;
    int numberSteps_;
    int index_;
    float payReceive_;
    float strike_;
    bool done_;

};


extern "C"
void cashFlowGeneratorCallerSwap(float* genFlows1, 
                                 float* genFlows2, 
                                 float* aux_data, 
                                 int paths, 
                                 int numberSteps,
                                 float* rates1, 
                                 float* rates2, 
                                 float* rates3, 
                                 float* forwards, 
                                 float* discRatios);

#endif


