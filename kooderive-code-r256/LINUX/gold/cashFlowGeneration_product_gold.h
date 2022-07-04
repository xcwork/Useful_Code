//
//                                                                          cashFlowGeneration_product_gold.h
//
//

// (c) Mark Joshi 2011,2014

#ifndef CASH_FLOW_GENERATION_PRODUCT_GOLD_H
#define CASH_FLOW_GENERATION_PRODUCT_GOLD_H
#include <vector>
#include <gold/MatrixFacade.h>
#include <gold/cashFlowGeneration_gold.h>
template<class D>
void cashFlowGeneratorCallerTARN_gold(std::vector<D>&  genFlows1, 
                                      std::vector<D>& genFlows2, 
                                      std::vector<D>& aux_data, 
                                      int paths, 
                                      int numberSteps,
                                      const   std::vector<D>&  rates1, 
                                      const   std::vector<D>&  rates2, 
                                      const   std::vector<D>&  rates3, 
                                      const   std::vector<D>&  forwards, 
                                      const   std::vector<D>&  discRatios);

template<class D>
void cashFlowGeneratorCallerSwap_gold(std::vector<D>& genFlows1, 
                                      std::vector<D>& genFlows2, 
                                      std::vector<D>& aux_data, 
                                      int paths, 
                                      int numberSteps,
                                      const   std::vector<D>&  rates1, 
                                      const   std::vector<D>&  rates2, 
                                      const   std::vector<D>&  rates3, 
                                      const   std::vector<D>&  forwards, 
                                      const   std::vector<D>&  discRatios);

template<class D>
class Swap_gold
{

    public:

        Swap_gold(const std::vector<D>& auxData, int numberSteps);
        Swap_gold( int numberSteps, D strike, D payReceive, const std::vector<D>& accruals1,
            const std::vector<D>& accruals2 );


        void newPath();

        int getIndex();

        bool getCashFlows(D& flow1, D& flow2, 
            D rate1, D rate2, D rate3, 
            const CubeConstFacade<D>& allForwardRates, const CubeConstFacade<D>& allDiscountRatios,  int paths );
        


    private:

        int numberSteps_;
        int index_;

        D strike_;
        D payReceive_;

        bool done_;

        std::vector<D> accruals1_;
        std::vector<D> accruals2_;

    };



template<class D>
    class TARN_gold
    {

    public:

        TARN_gold(const std::vector<D>& auxData, int numberSteps) : accruals1_(numberSteps), accruals2_(numberSteps)
        {
            numberSteps_ = numberSteps;
            totalCoupon_ = auxData[0];

            strike_ = auxData[1];
            multiplier_ =  auxData[2];

            for (int i=0; i < numberSteps; ++i)
            {
                accruals1_[i] = auxData[3+i];
                accruals2_[i] = auxData[3+numberSteps+i];
            }

            newPath();
        }

        void newPath()
        {
            index_=0;
            couponPaid_=0.0;
            done_ = false;
        }

        int getIndex()
        {
            return index_;
        }

        bool getCashFlows(D& flow1, D& flow2, D rate1, D rate2, D rate3, const CubeConstFacade<D>& allForwardRates, const CubeConstFacade<D>& allDiscountRatios,  int paths )
        {


            if (done_)
                return done_;

            D accrual2 = accruals2_[index_];

            flow2 = accrual2*rate2;

            if (index_ +1 == numberSteps_)
            {
                flow1 = couponPaid_-totalCoupon_ ;
                done_ = true;
                return done_;  
            }

            D accrual1 = accruals1_[index_];

            D uncappedFlow = (strike_-multiplier_*rate1)*accrual1;
            if (uncappedFlow <0.0)
                uncappedFlow =0.0f;

            D newTotalCoupon = couponPaid_ +uncappedFlow;
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

        int numberSteps_;
        int index_;

        D totalCoupon_;
        D strike_;
        D multiplier_;

        D couponPaid_;
        bool done_;

        std::vector<D> accruals1_;
        std::vector<D> accruals2_;

    };


template<class D>
void cashFlowGeneratorCallerTARN_gold(     std::vector<D>&  genFlows1, 
                                      std::vector<D>& genFlows2, 
                                      std::vector<D>& aux_data, 
                                      int paths, 
                                      int numberSteps,
                                      const   std::vector<D>&  rates1, 
                                      const   std::vector<D>&  rates2, 
                                      const   std::vector<D>&  rates3, 
                                      const   std::vector<D>&  forwards, 
                                      const   std::vector<D>&  discRatios)

{
    cashFlowGenerator_gold<TARN_gold<D>,D>(genFlows1,genFlows2,aux_data,paths,numberSteps,rates1,rates2,rates3,forwards,discRatios);
}


template<class D>
Swap_gold<D>::Swap_gold(const std::vector<D>& auxData, int numberSteps) : accruals1_(numberSteps), accruals2_(numberSteps)
{

    strike_ = auxData[0];
    payReceive_ =  auxData[1];

    for (int i=0; i < numberSteps; ++i)
    {
        accruals1_[i] = auxData[2+i];
        accruals2_[i] = auxData[2+numberSteps+i];
    }

    newPath();
}

template<class D>
Swap_gold<D>::Swap_gold( int numberSteps, D strike, D payReceive, const std::vector<D>& accruals1,
                     const std::vector<D>& accruals2 ) : strike_(strike), payReceive_(payReceive), accruals1_(accruals1),
                     accruals2_(accruals2)
{
        newPath();
}

template<class D>
void Swap_gold<D>::newPath()
{
    index_=0;

    done_ = false;
}
template<class D>
int Swap_gold<D>::getIndex()
{
    return index_;
}
template<class D>
bool  Swap_gold<D>::getCashFlows(D& flow1, D& flow2, D rate1, D rate2, D rate3, 
                              const CubeConstFacade<D>& allForwardRates, 
                              const CubeConstFacade<D>& allDiscountRatios,  int paths )
{

    D accrual2 = accruals2_[index_];
    flow2 = accrual2*rate1*payReceive_;


    D accrual1 = accruals1_[index_];
    flow1 = - accrual1*strike_*payReceive_;

    ++index_;

    done_ = (numberSteps_ == index_);

    return done_;

}



template<class D>
void cashFlowGeneratorCallerSwap_gold(    
                                      std::vector<D>&  genFlows1, 
                                      std::vector<D>& genFlows2, 
                                      std::vector<D>& aux_data, 
                                      int paths, 
                                      int numberSteps,
                                      const   std::vector<D>&  rates1, 
                                      const   std::vector<D>&  rates2, 
                                      const   std::vector<D>&  rates3, 
                                      const   std::vector<D>&  forwards, 
                                      const   std::vector<D>&  discRatios)

{
    cashFlowGenerator_gold<Swap_gold<D>,D>(genFlows1,genFlows2,aux_data,paths,numberSteps,rates1,rates2,rates3,forwards,discRatios);
}


#endif

