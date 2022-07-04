// (c) Mark Joshi 2012,2014
// This code is released under the GNU public licence version 3




#ifndef TRIGGERABLE_RANGE_ACCRUAL_GOLD_H
#define TRIGGERABLE_RANGE_ACCRUAL_GOLD_H
#include <gold/pragmas.h>
#include <vector>
#include <gold/MatrixFacade.h> 

/*
input paths should be paths \times steps
returnData will be paths \times numberAccrualPeriods
*/
template<class D>
void triggerableRangeAccrual_gold (const MatrixConstFacade<D>& inputPaths,
                                   int paths, 
                                   D triggerBarrier,
                                   D accrualBarrier,
                                   int numberAccrualPeriods,
                                   int stepsPerAccrualPeriod,
                                   int excessesRequiredPerTrigger,       
                                   std::vector<D>& returnData
                                   )
{
    if (inputPaths.columns() != stepsPerAccrualPeriod*numberAccrualPeriods)
        GenerateError("Input size mismatch in triggerableRangeAccrual_gold");

    returnData.resize(paths*numberAccrualPeriods);

    MatrixFacade<float> returnDataMatrix(&returnData[0],paths,numberAccrualPeriods);

    float maxCouponPerPeriod(0.0),rebate(0.0);

    float accrualPerDay = maxCouponPerPeriod/numberAccrualPeriods;

    for (int p=0; p < paths; ++p)
    {
        bool triggered=false;
        int stockStep=0; 
        for (int period=0; period < numberAccrualPeriods; ++period)
        {
            if (!triggered)
            {
                int accruedDays=0;
                int triggeringDays=0;
                for (int day=0; day < stepsPerAccrualPeriod; ++day, ++stockStep)
                {
                    D s= inputPaths(p,stockStep);
                    accruedDays += s < accrualBarrier ? 1 : 0;
                    triggeringDays += s > triggerBarrier ? 1 :0;

                }

                D coupon = accruedDays*accrualPerDay;

                if (triggeringDays >= excessesRequiredPerTrigger || period+1 == numberAccrualPeriods)
                {
                    triggered=true;
                    coupon += rebate;

                }

                returnDataMatrix(p,period) = coupon;
            }
            else
                 returnDataMatrix(p,period) =0.0f; 


        }
    }

}


#endif
